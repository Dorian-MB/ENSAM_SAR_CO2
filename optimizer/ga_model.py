from pathlib import Path
import sys
from pprint import pprint
if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
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
from optimizer.utils import ConfigBuilderFromSolution, LoggerForMultiprocessing, calculate_performance_metrics
from eco2_normandy.simulation import Simulation
from eco2_normandy.logger import Logger
from KPIS.utils import compute_dynamic_bounds 
from colorama import Fore

metrics_keys = ["cost", "wasted_production_over_time", "waiting_time", "underfill_rate"]

class SimulationProblem(ElementwiseProblem):
    """
    Problem definition for the simulation optimization using a genetic algorithm.
    """
    def __init__(self, base_config, boundaries: ConfigBoundaries,
                 multi_obj: bool = True, weights=None, kpis_list=None,
                 mertrics_keys=metrics_keys, elementwise_runner=None):
        """ Initialize the simulation problem, as a pymoo ElementwiseProblem.

        Args:
            base_config (_type_): _base configuration for the simulation.
            boundaries (ConfigBoundaries): 
            multi_obj (bool, optional): Defaults to True.
            weights (_type_, optional): Defaults to None.
        """
        self.base_config = base_config
        self.boundaries = boundaries
        self.max_ships = boundaries.max_num_ships
        self.multi_obj = multi_obj
        self.weights = weights or [20, 20, 15, 10]
        self.metrics_keys = mertrics_keys
        self.cfg_builder = ConfigBuilderFromSolution(base_config)
        self.kpis_list = kpis_list 

        # number of decision variables:
        # [num_storages, use_Bergen, use_Rotterdam, num_ship,
        #  ship_capacity, ship_speed]
        # + initial_destination_i (max_ships)
        # + fixed_storage_destination_i (max_ships)
        n_var = 6 + 2 * self.max_ships
        n_obj = 4 if multi_obj else 1

        # build lower/upper bounds
        xl = []
        xu = []
        # num_storages
        xl.append(1)
        xu.append(boundaries.max_num_storages)
        # use_Bergen, use_Rotterdam
        xl += [0, 0]
        xu += [1, 1]
        # num_ship
        xl.append(1)
        xu.append(boundaries.max_num_ships)
        # ship_capacity, ship_speed
        xl += [boundaries.ship_capacity_min, boundaries.ship_speed_min]
        xu += [boundaries.ship_capacity_max, boundaries.ship_speed_max]
        # initial destinations
        xl += [0] * self.max_ships
        xu += [boundaries.initial_destination] * self.max_ships
        # fixed storage destinations
        xl += [0] * self.max_ships
        xu += [boundaries.fixed_storage_destination] * self.max_ships

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=np.array(xl, dtype=float),
                         xu=np.array(xu, dtype=float),
                         elementwise=True,
                         elementwise_runner=elementwise_runner)

    def _run_simulation(self, cfg):
        """Run the simulation with animation."""
        sim = Simulation(config=cfg, verbose=False)
        sim.run()
        return sim

    def _evaluate(self, x, out, *args, **kwargs):
        # decode variables
        sol = self.cfg_builder.decode_and_repair(x, self.max_ships)
        # build simulation config
        cfg = self.cfg_builder.build(sol)
        sim = self._run_simulation(cfg)
        metrics = calculate_performance_metrics(cfg, sim, metrics_keys=self.metrics_keys)
        # assign objectives
        objectives = [metrics[k] for k in self.metrics_keys]
        self.kpis_list.append({k:o for k, o in zip(self.metrics_keys, objectives)})
        if not self.multi_obj:
            f = sum(w * v for w, v in zip(self.weights, objectives)) #todo: need to normalize 
            out['F'] = [f]
        else:
            out['F'] = objectives

class GAModel:
    def __init__(self, base_config, pop_size=20, n_gen=10, algorithm='NSGA3', 
                 weights=None, verbose=True, logger=None, 
                 parallelization:bool=False, n_pool:int=4,
                 *args, **kwargs):
        self.base_config = base_config
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.multi_obj = True if algorithm in ('NSGA3', 'NSGA2') else False
        self.n_obj = 4 if self.multi_obj else 1
        self.algorithm_name = algorithm
        self.weights = weights or [20, 20, 15, 30]
        self.verbose = verbose
        # logger cant be parallelized, so we use print
        self.log = (logger or Logger()) if not parallelization else LoggerForMultiprocessing()
        self.boundaries = ConfigBoundaries(logger=self.log)
        self.parallelization = parallelization
        self.n_pool = n_pool if parallelization else 1
        self.manager = Manager() if parallelization else None

        self.res = None
        self.algoritm = None
        self.cfg_builder = ConfigBuilderFromSolution(base_config)
        self.solutions = []
        self.scores = []
        self.istrain = False

        path = Path.cwd() / 'saved' / 'dynamic_bounds.csv'
        self.log.info(Fore.YELLOW+f"Loading absolute bounds from {Fore.CYAN+str(path.resolve())}"+Fore.RESET)
        if path.exists():
            self.log.info(Fore.GREEN+f"Using absolute bounds for normalization."+Fore.RESET)
            with open(path, 'r') as f:
                self.absolute_bounds = pd.read_csv(f, index_col="bounds").T.to_dict(orient='list')
        else:
            self.log.info(Fore.LIGHTRED_EX+f"Using dynamic bounds for normalization."+Fore.RESET)
            self.absolute_bounds = None

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
    
    def get_valid_pop_sizes(self):
        """Retourne les tailles de population valides pour NSGA3"""
        valid_sizes = self._calculate_valid_population_sizes(self.n_obj)
        return sorted([size[1] for size in valid_sizes])

    def _get_algorithm(self, algorithm):
        if self.algorithm_name == 'NSGA3':
            valid_pop_size = self._get_closest_valid_pop_size(self.pop_size, self.n_obj)
            if valid_pop_size != self.pop_size:
                self.log.warning(Fore.RED+f"Adjusting pop_size from {self.pop_size} to {valid_pop_size} for NSGA3"+Fore.RESET)
                self.pop_size = valid_pop_size
            ref_dirs = UniformReferenceDirectionFactory(n_dim=self.n_obj, n_points=self.pop_size).do()
            algorithm = NSGA3(pop_size=self.pop_size, eliminate_duplicates=True, ref_dirs=ref_dirs)
        elif self.algorithm_name == 'NSGA2':
            algorithm = NSGA2(pop_size=self.pop_size, eliminate_duplicates=True)
        elif self.algorithm_name == 'GA':
            algorithm = GA(pop_size=self.pop_size, eliminate_duplicates=True)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'NSGA3', 'NSGA2' or 'GA'.")
        return algorithm

    def solve(self, *args, **kwargs):

        if self.parallelization:
            pool = Pool(self.n_pool)  
            runner = StarmapParallelization(pool.starmap)
            kpis_list = self.manager.list()
            self.log.info(f"Using parallelization with {self.n_pool} processes.")
        else:
            self.log.info("Using single-threaded evaluation.")
            pool = None
            runner = LoopedElementwiseEvaluation() # lambda f, X: [f(x) for x in X]
            kpis_list = []

        self.problem = SimulationProblem(
            self.base_config, self.boundaries,
            multi_obj=self.multi_obj,
            weights=self.weights,
            kpis_list=kpis_list,
            elementwise_runner=runner
        )

        self.algorithm = self._get_algorithm(self.algorithm_name)
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
                verbose=True
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

        X = self.res.X
        F = self.res.F
        solutions = []
        results = []

        # Calculer les scores pondérés pour les solutions du front de Pareto
        F_norm = self.normalize(F)
        for idx in self.front_idx:
            score = sum(w * v for w, v in zip(self.weights, F_norm[idx]))
            x = X[idx]
            sol = self.cfg_builder.decode_and_repair(x, self.problem.max_ships)
            solutions.append({'solution'+str(idx): sol})
            results.append({f"results{idx}":F[idx].tolist() + [score]})

        idx, data = zip(*[(next(iter(s)), next(iter(s.values()))) for s in solutions])
        self.solutions = pd.DataFrame(data, index=idx)

        idx, data = zip(*[(next(iter(r)), next(iter(r.values()))) for r in results])
        self.scores = pd.DataFrame(data, index=idx, columns=self.problem.metrics_keys + ['score'])

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
        return pd.DataFrame([self.res.F[idx] for idx in self.front_idx])

    def _dynamic_normalize(self):
        abs_bounds = compute_dynamic_bounds(self.problem.kpis_list)
        return pd.DataFrame(abs_bounds, index=["min", "max"]).T.to_dict(orient='list')
        
    def normalize(self, F):
        if self.absolute_bounds:
            bounds = self.absolute_bounds
        else:
            bounds = self._dynamic_normalize()
        return [ [ (f-mn)/(mx - mn) for f, mx, mn in zip(fline, bounds['max'], bounds['min'])  ]
                for fline in F ] 

    def data_to_saved(self):
        if self.istrain is False:
            self.log.warning(Fore.RED+"No solutions or results available. Call solve() first."+Fore.RESET)
            return {}
        return {
            'solutions_ga': self.solutions,
            'scores_ga': self.scores,
            'pareto_ga': self.pareto_front,
        }
    
    @property
    def best_solution(self):
        return self._get_best()['solution']

    def _get_best(self): 
        """Retourne la meilleure solution (index dans le front de Pareto pour multi-obj)."""
        if self.res is None:
            raise RuntimeError('No results available. Call solve() first.')
        
        if self.multi_obj:
            # Trouver la meilleure solution du front de Pareto (somme pondérée minimale)
            F = self.res.F
            nds = NonDominatedSorting()
            front_indices = nds.do(F, only_non_dominated_front=True)

            # Trouver l'index de la meilleure solution (score minimal)
            scores = self.scores['score']
            best_front_idx = np.argmin(scores)
            best_idx = front_indices[best_front_idx]
            
            return {
                'solution': self.solutions.iloc[best_front_idx],
                'objectives': F[best_idx].tolist(),
                'score': scores.iloc[best_front_idx]
            }
        else:
            return {
                'solution': self.solutions.iloc[0],
                'objectives': self.res.F[0].tolist() if hasattr(self.res.F[0], '__iter__') else [self.res.F[0]],
                'score': self.res.F[0] if not hasattr(self.res.F[0], '__iter__') else self.res.F[0][0]
            }

    def log_score(self):
        if self.res is None:
            raise RuntimeError('No results available. Call solve() first.')
        
        best = self._get_best()
        objectives = {k: round(v,2) for k, v in zip(self.problem.metrics_keys, best['objectives'])}
        self.log.info(f"Best solution score {best['score']}")
        self.log.info(f"Objectives: {objectives}")
        self.log.info(f"Solution: {best['solution']}")

    def _run_simulation(self, cfg):
        """Run the simulation with the given configuration."""
        try:
            sim = Simulation(config=cfg, verbose=False)
            sim.run()
            return sim
        except Exception as e:
            self.log.error(Fore.RED+f"Simulation failed:"+Fore.RESET) 
            raise e

    def evaluate(self, cfg:dict, use_best_sol:bool=True, clip:bool=True):
        if self.istrain is False and use_best_sol is True:
            self.log.error(Fore.RED + "Model not trained yet. Please train the model before evaluating a configuration." + Fore.RESET)
            return None
        if use_best_sol:
            cfg = ConfigBuilderFromSolution(cfg).build(self.best_solution)
        sim = self._run_simulation(cfg)
        if sim is None:
            self.log.error(Fore.RED + f"Simulation {cfg.get('eval_name', '')} failed. Please check the configuration and try again." + Fore.RESET)
            return None
        metrics = calculate_performance_metrics(cfg, sim, metrics_keys=self.problem.metrics_keys)
        metrics["score"] =  sum( w * m for w, m in zip(self.normalize(), metrics.values()))
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
