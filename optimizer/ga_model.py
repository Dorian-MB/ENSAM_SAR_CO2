from pathlib import Path
import sys
from pprint import pprint
if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization, LoopedElementwiseEvaluation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from multiprocessing import Pool, Manager
from scipy.special import comb

from optimizer.boundaries import ConfigBoundaries
from optimizer.utils import (ConfigBuilderFromSolution, 
                            calculate_performance_metrics, 
                            Normalizer)
from eco2_normandy.simulation import Simulation
from eco2_normandy.logger import Logger
from KPIS.utils import compute_dynamic_bounds 
from colorama import Fore

metrics_keys = Normalizer().metrics_keys

class MixedVariableSampling(IntegerRandomSampling):
    """
    Custom sampling that understands variable types (Integer, Choice, Real).
    Generates samples respecting the semantic of each variable type.
    """
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
    
    def _do(self, problem, n_samples, **kwargs):
        # Generate samples for each variable type
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        
        for i, var_name in enumerate(problem.var_names):
            var = problem.variables[var_name]
            
            if isinstance(var, Integer):
                # Integer variables with bounds
                X[:, i] = np.random.randint(
                    var.bounds[0], 
                    var.bounds[1] + 1, 
                    size=n_samples
                )
            elif isinstance(var, Choice):
                # Choice variables - select from options
                X[:, i] = np.random.choice(
                    range(len(var.options)), 
                    size=n_samples
                )
            elif isinstance(var, Real):
                # Real variables (if any)
                X[:, i] = np.random.uniform(
                    var.bounds[0],
                    var.bounds[1],
                    size=n_samples
                ).astype(int)
        
        return X

class ShipConsistencyRepair(Repair):
    """
    Simplified repair operator leveraging ship._pick_new_destination() behavior.
    
    Key insight: When no fixed_storage_destination is set in the ship config, 
    ship._pick_new_destination() uses random.choice(self.storages), which automatically 
    handles storage availability. This allows for much simpler repair logic.
    """
    def __init__(self, max_ships, problem=None):
        super().__init__()
        self.max_ships = max_ships
        self.problem = problem
    
    def _do(self, problem, X, **kwargs):
        """
        Repair solutions with simplified logic:
        1. Force unused ships to default values (main efficiency gain)
        2. Set storage flags based on what's available in the scenario
        3. Minimal fixed destination repair (ship handles the rest automatically)
        
        Args:
            problem: The optimization problem
            X: Array of solutions (n_solutions x n_variables)
            
        Returns:
            X: Repaired solutions
        """
        # Store problem reference for variable name mapping
        if self.problem is None:
            self.problem = problem
            
            
        for i, x in enumerate(X):
            # Convert to dict for easier manipulation
            x_dict = problem._array_to_dict(x)
            num_ships = x_dict['num_ship']
            num_storages = x_dict['num_storages']

            # 1. Force unused ships to default values (main repair benefit)
            for ship_idx in range(num_ships, self.max_ships):
                x_dict[f'init{ship_idx+1}_destination'] = 0  # Le Havre
                x_dict[f'fixed{ship_idx+1}_storage_destination'] = 0  # Default

            # 2. Storage use fixed: not twice the same; and ensure at least one is used; and not more than available
            if num_storages == 2:
                x_dict['use_Bergen'] = 1
                x_dict['use_Rotterdam'] = 1
            elif not x_dict['use_Bergen'] and not x_dict['use_Rotterdam']:
                x_dict['use_Bergen'] = 0
                x_dict['use_Rotterdam'] = 1
            elif num_storages == 1 and x_dict['use_Bergen'] and x_dict['use_Rotterdam']:
                x_dict['use_Bergen'] = 0
                x_dict['use_Rotterdam'] = 1

            has_bergen = x_dict['use_Bergen'] == 1
            has_rotterdam = x_dict['use_Rotterdam'] == 1

            # 3. Simple fixed destination repair - only fix obvious conflicts
            # map_ship_fixed_storage_destination = {0: "Rotterdam", 1: "Bergen"}
            # map_ship_initial_destination = {0: "Le Havre", 1: "Rotterdam", 2: "Bergen"}
            for ship_idx in range(num_ships): 
                current_fixed = x_dict[f'fixed{ship_idx+1}_storage_destination']
                current_init = x_dict[f'init{ship_idx+1}_destination']

                # Only repair if the chosen storage is completely unavailable
                if current_fixed == 1 and not has_bergen:  # Wants Bergen but unavailable
                    x_dict[f'fixed{ship_idx+1}_storage_destination'] = 0  # → Rotterdam
                elif current_fixed == 0 and not has_rotterdam:  # Wants Rotterdam but unavailable  
                    x_dict[f'fixed{ship_idx+1}_storage_destination'] = 1  # → Bergen

                # If the initial destination is unavailable, set it to the factory
                if current_init == 2 and not has_bergen:  # Wants Bergen but unavailable
                    x_dict[f'init{ship_idx+1}_destination'] = 0  
                elif current_init == 1 and not has_rotterdam:  # Wants Rotterdam but unavailable
                    x_dict[f'init{ship_idx+1}_destination'] = 0 

            # Convert back to array
            X[i] = problem._dict_to_array(x_dict)
        
        return X

class SimulationProblem(ElementwiseProblem):
    """Problem definition for the simulation optimization using a genetic algorithm."""
    def __init__(self, base_config, boundaries: ConfigBoundaries, logger=None,
                 multi_obj: bool = True, weights=None, kpis_list=None,
                 mertrics_keys=metrics_keys, caps_steps=1000, elementwise_runner=None):
        """Initialize the simulation problem, as a pymoo ElementwiseProblem with discrete variables."""
        self.log = logger or Logger()
        self.base_config = base_config
        self.boundaries = boundaries
        self.max_ships = boundaries.max_num_ships
        self.multi_obj = multi_obj
        self.caps_steps = caps_steps
        # default weights from Normalizer for consistency across modules
        self.weights = weights or Normalizer().metrics_weight
        self.metrics_keys = mertrics_keys
        self.cfg_builder = ConfigBuilderFromSolution(base_config, boundaries)
        self.kpis_list = kpis_list 

        # Define variables with proper types and automatic steps
        self.variables = {}
        self.var_names = []
        
        # Core variables with discrete types
        self.variables['num_storages'] = Integer(bounds=(1, boundaries.max_num_storages))
        self.variables['use_Bergen'] = Choice(options=[0, 1])
        self.variables['use_Rotterdam'] = Choice(options=[0, 1]) 
        self.variables['num_ship'] = Integer(bounds=(1, boundaries.max_num_ships))
        self.variables['ship_speed'] = Integer(bounds=(
            boundaries.ship_speed["min"],
            boundaries.ship_speed["max"] 
        ))
        self.variables['number_of_tanks'] = Integer(bounds=(
            boundaries.factory_tanks["min"],
            boundaries.factory_tanks["max"]
        ))
        
        # Variables with automatic steps
        self.variables['ship_capacity'] = Integer(bounds=(
            boundaries.ship_capacity["min"] // caps_steps,
            boundaries.ship_capacity["max"] // caps_steps
        ))
        self.variables['storage_caps'] = Integer(bounds=(
            boundaries.storage_caps["min"] // caps_steps,
            boundaries.storage_caps["max"] // caps_steps
        ))

        
        # Variables per ship (destinations) with explicit choices
        for i in range(boundaries.max_num_ships):
            self.variables[f'init{i+1}_destination'] = Choice(options=[0, 1, 2])  # Le Havre, Rotterdam, Bergen
            self.variables[f'fixed{i+1}_storage_destination'] = Choice(options=[0, 1])  # Rotterdam, Bergen

        # Store variable names in order for array conversion
        self.var_names = list(self.variables.keys())
        n_var = len(self.var_names)
        n_obj = 4 if multi_obj else 1
        
        # Create bounds arrays from variables (for ElementwiseProblem compatibility)
        xl = []
        xu = []
        for var_name in self.var_names:
            var = self.variables[var_name]
            if isinstance(var, (Integer, Real)):
                xl.append(var.bounds[0])
                xu.append(var.bounds[1])
            elif isinstance(var, Choice):
                xl.append(0)
                xu.append(len(var.options) - 1)
        
        super().__init__(n_var=n_var, n_obj=n_obj, 
                        xl=np.array(xl, dtype=int), 
                        xu=np.array(xu, dtype=int),
                        elementwise=True, elementwise_runner=elementwise_runner)
    
    def _array_to_dict(self, x):
        """Convert array solution to dictionary, applying steps and choice mapping"""
        x_dict = {}
        for name, val in zip(self.var_names, x, strict=True):
            var = self.variables[name]
            
            if isinstance(var, Integer):
                # Apply steps for capacity variables
                if name in ['ship_capacity', 'storage_caps']:
                    x_dict[name] = int(val) * self.caps_steps
                else:
                    x_dict[name] = int(val)
            elif isinstance(var, Choice):
                # Map choice index to actual option value
                x_dict[name] = var.options[int(val)]
            elif isinstance(var, Real):
                x_dict[name] = float(val)
            else:
                x_dict[name] = int(val)
        
        return x_dict
    
    def _dict_to_array(self, x_dict):
        """Convert dictionary solution back to array, reversing steps and choice mapping"""
        x_array = []
        for name in self.var_names:
            var = self.variables[name]
            val = x_dict[name]
            
            if isinstance(var, Integer):
                # Reverse steps for capacity variables
                if name in ['ship_capacity', 'storage_caps']:
                    x_array.append(val // self.caps_steps)
                else:
                    x_array.append(int(val))
            elif isinstance(var, Choice):
                # Map option value back to choice index
                x_array.append(var.options.index(val))
            elif isinstance(var, Real):
                x_array.append(float(val))
            else:
                x_array.append(int(val))
        
        return np.array(x_array)

    def _run_simulation(self, cfg):
        """Run the simulation with animation."""
        sim = Simulation(config=cfg, verbose=False)
        try: 
            sim.run()
        except Exception as e:
            self.log.error(Fore.RED+f"Simulation failed, config: {cfg}"+Fore.RESET)
            raise e
        return sim

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert array to dictionary (steps are applied automatically)
        x_dict = self._array_to_dict(x)
        
        # Build simulation config directly from variable dict  
        cfg = self.cfg_builder.build(x_dict)
        sim = self._run_simulation(cfg)
        metrics = calculate_performance_metrics(cfg, sim, metrics_keys=self.metrics_keys)
        
        # Assign objectives
        objectives = [metrics[k].iloc[0] for k in self.metrics_keys]
        if hasattr(self.kpis_list, 'append'):
            self.kpis_list.append(metrics)
        
        if not self.multi_obj:
            f = sum(w * v for w, v in zip(self.weights, objectives))
            out['F'] = [f]
        else:
            out['F'] = objectives

class GAModel:
    def __init__(self, base_config, pop_size=20, n_gen=10, algorithm='NSGA3',
                 weights=None, verbose=True, logger=None,
                 parallelization: bool = False, n_pool: int = 4,
                 caps_steps=1000, *args, **kwargs):
        self.base_config = base_config
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.multi_obj = True if algorithm in ('NSGA3', 'NSGA2') else False
        self.n_obj = 4 if self.multi_obj else 1
        self.algorithm_name = algorithm
        self.caps_steps = caps_steps
        # default weights from Normalizer for consistency across modules
        self.weights = weights or Normalizer().metrics_weight
        self.verbose = verbose
        # logger cant be parallelized, so we use print
        self.log = logger or Logger()
        self.parallelization = parallelization
        self.n_pool = n_pool if parallelization else 1
        self.manager = Manager() if parallelization else None

        self.res = None
        self.algoritm = None
        self.solutions = []
        self.scores = []
        self.istrain = False

        self.boundaries = ConfigBoundaries(logger=self.log)
        self.cfg_builder = ConfigBuilderFromSolution(base_config, self.boundaries)
        self.normalize = Normalizer()

    def _calculate_valid_population_sizes(self, n_dim, max_size=500):
        """Calcule les tailles de population valides pour NSGA3"""
        valid_sizes = []
        for n_partitions in range(1, 20):  # test jusqu'à 20 partitions
            n_points = int(comb(n_partitions + n_dim - 1, n_dim - 1))
            if n_points > max_size:
                break
            valid_sizes.append((n_partitions, n_points))
        return valid_sizes

    def _get_closest_valid_pop_size(self, desired_size, n_dim):
        """Trouve la taille de population valide la plus proche"""
        valid_sizes = self._calculate_valid_population_sizes(n_dim)
        closest = min(valid_sizes, key=lambda x: abs(x[1] - desired_size))
        return closest[1]
    
    def _get_valid_pop_sizes(self):
        """Retourne les tailles de population valides pour NSGA3"""
        valid_sizes = self._calculate_valid_population_sizes(self.n_obj)
        return sorted([size[1] for size in valid_sizes])

    def _get_algorithm(self, sampling):
        # Create ship consistency repair operator
        repair = ShipConsistencyRepair(self.boundaries.max_num_ships)
        
        if self.algorithm_name == 'NSGA3':
            valid_pop_size = self._get_closest_valid_pop_size(self.pop_size, self.n_obj)
            if valid_pop_size != self.pop_size:
                self.log.warning(Fore.RED+f"Adjusting pop_size from {self.pop_size} to {valid_pop_size} for NSGA3"+Fore.RESET)
                self.pop_size = valid_pop_size
            ref_dirs = UniformReferenceDirectionFactory(n_dim=self.n_obj, n_points=self.pop_size).do()
            algorithm = NSGA3(
                pop_size=self.pop_size, 
                eliminate_duplicates=True, 
                ref_dirs=ref_dirs, 
                repair=repair,
                sampling=sampling
            )
        elif self.algorithm_name == 'NSGA2':
            algorithm = NSGA2(
                pop_size=self.pop_size, 
                eliminate_duplicates=True, 
                repair=repair,
                sampling=sampling
            )
        elif self.algorithm_name == 'GA':
            algorithm = GA(
                pop_size=self.pop_size, 
                eliminate_duplicates=True, 
                repair=repair,
                sampling=sampling
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'NSGA3', 'NSGA2' or 'GA'.")
        return algorithm

    def solve(self, *args, **kwargs):
        if self.parallelization:
            pool = Pool(self.n_pool)  
            runner = StarmapParallelization(pool.starmap)
            kpis_list = self.manager.list()
            if self.verbose: self.log.info(f"Using parallelization with {self.n_pool} processes.")
        else:
            if self.verbose: self.log.info("Using single-threaded evaluation.")
            pool = None
            runner = LoopedElementwiseEvaluation() # lambda f, X: [f(x) for x in X]
            kpis_list = []

        self.problem = SimulationProblem(
            self.base_config, self.boundaries,
            multi_obj=self.multi_obj,
            weights=self.weights,
            kpis_list=kpis_list,
            caps_steps=self.caps_steps,
            elementwise_runner=runner,
            logger=self.log
        )

        sampling = MixedVariableSampling(self.problem)
        self.algorithm = self._get_algorithm(sampling)

        # Create mixed variable sampling with problem context
        # Update algorithm with proper sampling
        # self.algorithm.sampling = sampling
        
        self.res = self._minimize(self.algorithm, pool)
        if self.parallelization: 
            pool.close() 
            pool.join()
            self.problem.kpis_list = list(kpis_list)  # Convertir la liste partagée en liste normale
        self._set_results()

    def _minimize(self, algo, pool):
        """
        Minimize the problem using the given algorithm.
        handle errors and parallelization. close the pool even if an error occurs.
        """
        try:
            termination = get_termination('n_gen', self.n_gen)
            self.res = minimize(
                self.problem,
                algo,
                termination,
                seed=1,
                verbose=self.verbose
            )
            self.istrain = True
        except Exception as e:
            self.log.error(Fore.RED+f"Error occurred during minimization `GASolver`: {e}"+Fore.RESET)
            raise e
        finally:
            if self.parallelization:
                pool.close()
        return self.res

    def _set_results(self):
        if self.istrain is False:
            raise RuntimeError('No results available. Call solve() first.')

        X = self.res.X[self.front_idx]
        F = self.res.F[self.front_idx]
        F = pd.DataFrame(F, columns=self.problem.metrics_keys, index=[f"solution_{i}" for i in range(len(F))]) # index important in case of singleton
        self.scores = F.copy()
        solutions = []

        F_norm = self.normalize(F)
        self.scores["score"] = self.normalize.compute_score(F_norm)
        self.scores.index.name = 'solution_id'

        for idx, x in enumerate(X):
            # Convert array to dictionary (steps applied automatically)
            x_dict = self.problem._array_to_dict(x)
            solutions.append({'solution_'+str(idx): x_dict})

        idx, data = zip(*[(next(iter(s)), next(iter(s.values()))) for s in solutions])
        self.solutions = pd.DataFrame(data, index=idx)
        self.solutions.index.name = 'solution_id'

    @property
    def front_idx(self):
        if not self.istrain:
            raise RuntimeError('No results available. Call solve() first.')
        F = self.res.F
        if self.multi_obj:
            nds = NonDominatedSorting()
            front = nds.do(F, only_non_dominated_front=True)
        else:
            front = [int(np.argmin(F[:, 0]))]
        return front

    @property
    def pareto_front(self):
        if not self.istrain:
            raise RuntimeError('No results available. Call solve() first.')
        F = self.res.F[self.front_idx]
        return pd.DataFrame(F, columns=self.problem.metrics_keys, index=[f"solution_{i}" for i in range(len(F))])

    @property
    def best_score(self):
        return self.scores.loc[self.scores["score"].idxmin()]
    
    @property
    def best_solution(self):
        return self.solutions.loc[self.best_score.name]

    def log_score(self):
        if self.res is None:
            raise RuntimeError('No results available. Call solve() first.')

        self.log.info(Fore.LIGHTCYAN_EX + f"Best solution score {Fore.RESET}{self.best_score['score']}")
        self.log.info(Fore.LIGHTCYAN_EX + f"Objectives:\n{Fore.RESET}{self.best_score.drop('score')}")
        self.log.info(Fore.LIGHTCYAN_EX + f"Solution:\n{Fore.RESET}{self.best_solution}")

    def _run_simulation(self, cfg):
        """Run the simulation with the given configuration."""
        try:
            sim = Simulation(config=cfg, verbose=False)
            sim.run()
            return sim
        except Exception as e:
            self.log.error(Fore.RED+f"Simulation failed:"+Fore.RESET) 
            raise e

    def data_to_saved(self):
        if self.istrain is False:
            self.log.warning(Fore.RED+"No solutions or results available. Call solve() first."+Fore.RESET)
            return {}
        return {
            'solutions_ga': self.solutions,
            'scores_ga': self.scores,
            'pareto_ga': self.pareto_front,
        }
        
    def evaluate(self, cfg:dict, clip:bool=True):
        if self.istrain is False:
            self.log.error(Fore.RED + "Model not trained yet. Please train the model before evaluating a configuration." + Fore.RESET)
            return None
        cfg = ConfigBuilderFromSolution(cfg, self.boundaries).build(self.best_solution)
        sim = self._run_simulation(cfg)
        if sim is None:
            self.log.error(Fore.RED + f"Simulation {cfg.get('eval_name', '')} failed. Please check the configuration and try again." + Fore.RESET)
            return None
        metrics = calculate_performance_metrics(cfg, sim, metrics_keys=self.problem.metrics_keys)
        normed_metrics = self.normalize(metrics)
        metrics["score"] = self.normalize.compute_score(normed_metrics)
        return metrics


if __name__ == '__main__':
    # example usage
    from eco2_normandy.tools import get_simlulation_variable
    path = 'scenarios/dev/phase3_bergen_18k_2boats.yaml'
    base_cfg = get_simlulation_variable(path)[0]
    solver = GAModel(base_cfg, pop_size=100, n_gen=10, parallelization=True)
    res = solver.solve()
    # for r in solver.get_results():
    #     print(r)
