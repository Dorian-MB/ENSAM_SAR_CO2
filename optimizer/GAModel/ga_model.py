from pathlib import Path
import sys
if __name__ == "__main__":
    sys.path.insert(0, str(object=Path.cwd()))

import numpy as np
import pandas as pd

import dill

from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling

# from pymoo.operators.sampling.rnd import IntegerRandomSampling
# from pymoo.operators.mutation.pm import PolynomialMutation
# from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.optimize import minimize
from pymoo.core.result import Result
from pymoo.core.problem import LoopedElementwiseEvaluation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from multiprocessing import Manager
from scipy.special import comb

from optimizer.boundaries import ConfigBoundaries
from optimizer.utils import (
    ConfigBuilderFromSolution,
    calculate_performance_metrics,
    Normalizer,
)
from eco2_normandy.simulation import Simulation
from eco2_normandy.logger import Logger
from colorama import Fore
from optimizer.GAModel.utils import ShipConsistencyRepair, MixedVariableSampling, SerializableStarmapRunner
from optimizer.GAModel.problem import SimulationProblem

metrics_keys = Normalizer().metrics_keys



class GaModel:
    def __init__(
        self,
        base_config: dict,
        pop_size: int | None = 20,
        n_gen: int | None = 10,
        algorithm_name: str | None = "NSGA3",
        verbose: bool | int = True,
        logger: Logger | None = None,
        parallelization: bool = False,
        n_pool: int | None = None,
        caps_steps: int | None = 1000,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the Genetic Algorithm model.

        Args:
            base_config (dict): The base configuration for the model.
            pop_size (int | None, optional): The population size. Defaults to 20.
            n_gen (int | None, optional): The number of generations. Defaults to 10.
            algorithm_name (str | None, optional): The name of the algorithm to use. Defaults to "NSGA3".
            verbose (bool | int, optional): Whether to print verbose output. Defaults to True.
            logger (Logger | None, optional): The logger to use. Defaults to None.
            parallelization (bool, optional): Whether to use parallelization. Defaults to False.
            n_pool (int | None, optional): The number of processes to use for parallelization. Defaults to None (None: auto detect optimal(maximum)).
            caps_steps (int | None, optional): The number of steps to use for the caps. Defaults to 1000.
        """
        self.base_config = base_config
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_obj = 4
        self.algorithm_name = algorithm_name
        self.algorithm = None
        self.caps_steps = caps_steps
        self.verbose = verbose
        self.log = logger or Logger()
        self.parallelization = parallelization
        self.init_parallelization = parallelization
        self.runner = None
        self.kpis_list = Manager().list() if parallelization else []
        self.n_pool = n_pool if parallelization else 1
        self.manager = Manager() if parallelization else None

        self.res = None
        self.solutions = []
        self.scores = []
        self.istrain = False
        self.from_dill = False

        self.boundaries = ConfigBoundaries(logger=self.log, verbose=0)
        self.cfg_builder = ConfigBuilderFromSolution(base_config, self.boundaries)
        self.normalize = Normalizer()
        self.problem = SimulationProblem(base_config, self.boundaries)
        self.metrics_keys = self.normalize.metrics_keys

    def reset(self, base_config: dict) -> None:
        self.res = None
        self.solutions = []
        self.scores = []
        self.istrain = False
        self.base_config = base_config
        self.cfg_builder = ConfigBuilderFromSolution(base_config, self.boundaries)
        self.kpis_list = self.manager.list() if self.parallelization else []

    def _calculate_valid_population_sizes(self, n_dim: int, max_size: int = 500) -> list[tuple[int, int]]:
        """Calcule les tailles de population valides pour NSGA3"""
        valid_sizes = []
        for n_partitions in range(1, 20):  # test jusqu'à 20 partitions
            n_points = int(comb(n_partitions + n_dim - 1, n_dim - 1))
            if n_points > max_size:
                break
            valid_sizes.append((n_partitions, n_points))
        return valid_sizes

    def _get_closest_valid_pop_size(self, desired_size: int, n_dim: int) -> int:
        """Trouve la taille de population valide la plus proche"""
        valid_sizes = self._calculate_valid_population_sizes(n_dim)
        closest = min(valid_sizes, key=lambda x: abs(x[1] - desired_size))
        return closest[1]

    def _get_valid_pop_sizes(self) -> list[int]:
        """Retourne les tailles de population valides pour NSGA3"""
        valid_sizes = self._calculate_valid_population_sizes(self.n_obj)
        return sorted([size[1] for size in valid_sizes])

    def _get_algorithm(self, repair: Repair, sampling: Sampling = None) -> GeneticAlgorithm:
        # ? Check algo: https://pymoo.org/algorithms/list.html
        # Create ship consistency repair operator
        if sampling is None:
            sampling = FloatRandomSampling()
        if self.algorithm_name == "NSGA3":
            valid_pop_size = self._get_closest_valid_pop_size(self.pop_size, self.n_obj)
            if valid_pop_size != self.pop_size:
                self.log.warning(
                    Fore.RED + f"Adjusting pop_size from {self.pop_size} to {valid_pop_size} for NSGA3" + Fore.RESET
                )
                self.pop_size = valid_pop_size
            ref_dirs = UniformReferenceDirectionFactory(n_dim=self.n_obj, n_points=self.pop_size).do()
            algorithm = NSGA3(
                pop_size=self.pop_size,
                eliminate_duplicates=True,
                ref_dirs=ref_dirs,
                repair=repair,
                sampling=sampling,
            )
        elif self.algorithm_name == "NSGA2":
            algorithm = NSGA2(
                pop_size=self.pop_size,
                eliminate_duplicates=True,
                repair=repair,
                sampling=sampling,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'NSGA3', 'NSGA2'.")
        return algorithm

    def solve(self, *args, **kwargs) -> None:
        if self.init_parallelization:
            self.runner = SerializableStarmapRunner(self.n_pool)
            self.init_parallelization = False
            if self.verbose:
                self.log.info(f"Using parallelization with {self.runner.n_processes} processes.")
        elif not self.parallelization:
            if self.verbose:
                self.log.info("Using single-threaded evaluation.")
            self.runner = LoopedElementwiseEvaluation()

        self.problem = SimulationProblem(
            self.base_config,
            self.boundaries,
            kpis_list=self.kpis_list,
            caps_steps=self.caps_steps,
            elementwise_runner=self.runner,
            logger=self.log,
        )

        if not self.from_dill:
            repair = ShipConsistencyRepair(self.boundaries.max_num_ships)
            sampling = MixedVariableSampling()
            self.algorithm = self._get_algorithm(repair, sampling)


        self._minimize(self.algorithm, keep_alive=kwargs.get("keep_alive", False))

        if self.parallelization:
            self.problem.kpis_list = list(self.kpis_list)
            self.kpis_list = list(self.kpis_list)
        self._set_results()

    def _minimize(self, algo: GeneticAlgorithm, *args, **kwargs) -> Result:
        """
        Minimize the problem using the given algorithm.
        handle errors and parallelization. close the pool even if an error occurs.
        """
        try:
            self.algorithm.termination = MaximumGenerationTermination(self.n_gen)

            self.res = minimize(
                self.problem,
                algo,
                seed=42,
                verbose=self.verbose,
                copy_algorithm=False,
                save_history=True,
            )
            self.istrain = True

        except Exception as e:
            if self.parallelization:
                self.runner.close()
            self.log.error(Fore.RED + f"Error occurred during minimization `GASolver`: {e}" + Fore.RESET)
            raise e
        if self.parallelization and not kwargs.get("keep_alive", False):
            self.runner.close()
        return self.res

    def _set_results(self) -> None:
        if self.istrain is False:
            raise RuntimeError("No results available. Call solve() first.")

        front_idx = self._front_idx
        X = self.res.X[front_idx]
        F = self.res.F[front_idx]
        F = pd.DataFrame(
            F,
            columns=self.metrics_keys,
            index=[f"solution_{i}" for i in range(len(F))],
        )  # index important in case of singleton
        self.scores = F.copy()
        solutions = []

        F_norm = self.normalize(F)
        self.scores["score"] = self.normalize.compute_score(F_norm)
        self.scores.index.name = "solution_id"

        for idx, x in enumerate(X):
            # Convert array to dictionary (steps applied automatically)
            x_dict = self.problem._array_to_dict(x)
            solutions.append({"solution_" + str(idx): x_dict})

        idx, data = zip(*[(next(iter(s)), next(iter(s.values()))) for s in solutions])
        self.solutions = pd.DataFrame(data, index=idx)
        self.solutions.index.name = "solution_id"

        self.pareto_front = pd.DataFrame(
            F,
            columns=self.metrics_keys,
            index=[f"solution_{i}" for i in range(len(F))],
        )
        self.pareto_front.index.name = "solution_id"

    @property
    def _front_idx(self) -> np.ndarray:
        if not self.istrain:
            raise RuntimeError("No results available. Call solve() first.")
        F = self.res.F
        nds = NonDominatedSorting()
        front = nds.do(F, only_non_dominated_front=True)
        return front

    @property
    def best_score(self) -> pd.Series:
        return self.scores.loc[self.scores["score"].idxmin()]

    @property
    def best_solution(self) -> pd.Series:
        return self.solutions.loc[self.best_score.name]

    def get_best_simulation(self, num_period: int = 2000) -> tuple[Simulation, dict]:
        """Get the best simulation based on the current optimization results.

        Args:
            num_period (int, optional): Number of periods for the simulation. Defaults to 2000.

        Returns:
            tuple[Simulation, dict]: The best simulation and its configuration.
        """
        if self.istrain is False:
            raise RuntimeError("No results available. Call solve() first.")
        best_cfg = self.cfg_builder.build(self.best_solution, num_period=num_period)
        sim = self._run_simulation(best_cfg)
        return sim, best_cfg

    def log_score(self) -> None:
        if self.res is None:
            raise RuntimeError("No results available. Call solve() first.")

        self.log.info(Fore.LIGHTCYAN_EX + f"Best solution score {Fore.RESET}{self.best_score['score']}")
        self.log.info(Fore.LIGHTCYAN_EX + f"Objectives:\n{Fore.RESET}{self.best_score.drop('score')}")
        self.log.info(Fore.LIGHTCYAN_EX + f"Solution:\n{Fore.RESET}{self.best_solution}")

    def _run_simulation(self, cfg: dict) -> Simulation:
        """Run the simulation with the given configuration."""
        try:
            sim = Simulation(config=cfg, verbose=False)
            sim.run()
            return sim
        except Exception as e:
            self.log.error(Fore.RED + f"Simulation failed:" + Fore.RESET)
            raise e

    def data_to_saved(self) -> dict:
        if self.istrain is False:
            self.log.warning(Fore.RED + "No solutions or results available. Call solve() first." + Fore.RESET)
            return {}
        name = f"{self.algorithm_name}"
        return {
            f"{name}_pareto_front": self.pareto_front,
            f"{name}_all_solutions": self.solutions,
            f"{name}_all_scores": self.scores,
            f"model_{name}": self.algorithm,
            f"{name}_results": self.res,
        }

    def evaluate(self, cfg: dict, clip: bool = True) -> pd.DataFrame:
        if self.istrain is False:
            self.log.error(
                Fore.RED
                + "Model not trained yet. Please train the model before evaluating a configuration."
                + Fore.RESET
            )
            raise RuntimeError("Model not trained.")
        cfg = ConfigBuilderFromSolution(cfg, self.boundaries).build(self.best_solution)
        sim = self._run_simulation(cfg)
        if sim is None:
            self.log.error(
                Fore.RED
                + f"Simulation {cfg.get('eval_name', '')} failed. Please check the configuration and try again."
                + Fore.RESET
            )
            raise RuntimeError("Simulation failed during evaluation.")
        metrics = calculate_performance_metrics(cfg, sim, metrics_keys=self.problem.metrics_keys)
        normed_metrics = self.normalize(metrics, clip=clip)
        metrics["score"] = self.normalize.compute_score(normed_metrics)
        return metrics

    @classmethod
    def load(
        cls,
        sol_dir_path: str | Path,
        base_config: dict,
        pop_size: int | None = 20,
        n_gen: int | None = 10,
        algorithm_name: str | None = None,
        verbose: bool | int = True,
        logger: Logger | None = None,
        parallelization: bool = False,
        n_pool: int | None = None,
        caps_steps: int | None = 1000,
        **kwargs,
    ) -> "GaModel":
        """Load solutions from a NPY file"""
        log = logger or Logger()
        if isinstance(sol_dir_path, str):
            sol_dir_path = Path(sol_dir_path)
        if not sol_dir_path.exists():
            log.error(Fore.RED + f"Solution {sol_dir_path} not found." + Fore.RESET)
            raise FileNotFoundError(f"Solution {sol_dir_path} not found.")
        if not sol_dir_path.is_dir():
            raise NotADirectoryError(f"Solution path {sol_dir_path} is not a directory.")

        dill_file = list(sol_dir_path.glob("*.dill"))
        if not dill_file:
            log.error(Fore.RED + f"No DILL files found in {sol_dir_path}." + Fore.RESET)
            raise FileNotFoundError(f"No DILL files found in {sol_dir_path}.")
        elif len(dill_file) != 2:
            log.error(Fore.RED + f"Expected 2 DILL files, found {len(dill_file)}." + Fore.RESET)
            raise ValueError(f"Expected 2 DILL files, found {len(dill_file)}.")
        for dill_path in dill_file:
            with open(dill_path, "rb") as f:
                if "model" in dill_path.name:
                    algorithm = dill.load(f)
                elif "results" in dill_path.name:
                    res = dill.load(f)
                else:
                    log.error(
                        Fore.RED + f"Unknown DILL file {dill_path.name}. Expected 'model' or 'results' files" + Fore.RESET
                    )
                    raise ValueError(f"Unknown DILL file {dill_path.name}. Expected 'model' or 'results' files")

        log.info(Fore.GREEN + f"Model Loaded from {Fore.LIGHTCYAN_EX}{sol_dir_path.resolve()}" + Fore.RESET)

        # Create instance with loaded data
        instance = cls(
            base_config=base_config,
            pop_size=pop_size,
            n_gen=n_gen,
            algorithm_name=algorithm_name or ("NSGA3" if "NSGA3" in str(sol_dir_path.name) else "NSGA2"),
            verbose=verbose,
            logger=log,
            parallelization=parallelization,
            n_pool=n_pool,
            caps_steps=caps_steps,
        )

        # Set loaded data
        instance.algorithm = algorithm
        instance.res = res
        instance.istrain = True
        instance.from_dill = True
        instance._set_results()

        return instance


if __name__ == "__main__":
    # example usage
    from eco2_normandy.tools import get_simlulation_variable

    path = "scenarios/dev/phase3_bergen_18k_2boats.yaml"
    base_cfg = get_simlulation_variable(path)[0]
    solver = GaModel(base_cfg, pop_size=10, n_gen=2, parallelization=True)
    res = solver.solve()
