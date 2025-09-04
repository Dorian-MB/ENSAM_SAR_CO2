import time
import cProfile, pstats
import os
import sys
from pathlib import Path
import random

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import dill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from colorama import Fore
from typing import Generator
from pymoo.core.result import Result

from KPIS import Kpis
from eco2_normandy.logger import Logger
from optimizer.utils import (
    get_all_scenarios,
    NoProfiler,
    evaluate_single_scenario,
)
from optimizer.compare_scenarios import print_diffs
from optimizer.boundaries import ConfigBoundaries
from optimizer.GAModel.history_analyzer import NSGA3HistoryAnalyzer
from optimizer.GAModel.ga_model import GaModel
from optimizer.CPModel.cp_model import CpModel


class OptimizationOrchestrator:
    """
    Base class for optimization algorithms.
    """

    def __init__(
        self,
        model,
        logger: Logger | None = None,
        verbose: int | bool = 1,
        enable_cprofile: bool = False,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        self.log = logger or (model.log if model.log else Logger())
        self.model = model
        self.verbose = verbose
        self.enable_cprofile = enable_cprofile
        self.profiler = None
        self.boundaries = ConfigBoundaries(verbose=0)
        self.histories = {}
        self.seed_it_all(seed)

    @staticmethod
    def seed_it_all(seed=42) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def compare_solution_to_base_config(self, solution: dict = None) -> None:
        if solution is None:
            solution = self.model.best_solution
        sol_cfg = self.model.cfg_builder.build(solution)
        base_config = self.model.base_config
        sol_cfg["name"] = "model_solution"
        base_config["name"] = "base_config"
        print_diffs(sol_cfg, base_config)

    @staticmethod
    def _get_scenarios(
        path: str = "scenarios/", scenario_filter: str = "phase"
    ) -> Generator[tuple[Path, dict], None, None]:
        """
        Récupère tous les scénarios à partir d'un fichier YAML.
        """
        for s_path, scenario in get_all_scenarios(path):
            if scenario_filter and scenario_filter not in str(s_path.parent):
                continue
            yield s_path, scenario

    def evaluate_all_scenarios(
        self,
        num_period: int = 2000,
        path: str = "scenarios/",
        scenario_filter: str = "phase",
    ) -> pd.DataFrame:
        """Evalue all scenarios in the given path.

        Args:
            num_period (int, optional): number of simulation loop. Defaults to 2000.
            path (str, optional): path of scenarios. Defaults to "scenarios/".
            use_best_sol (bool, optional): whether to use the best solution for evaluation. Defaults to True. False allow to use default scenario.
            scenario_filter (str, optional): filter for scenarios directory. Defaults to "phase".
            limite (int, optional): limit of scenario tested, None:test all scenarios found. Defaults to None.

        Returns:
            pd.DataFrame: MultiIndex DataFrame with results of the evaluation.
        """
        if self.model.istrain is False:
            self.log.info("model not trained yet, `Optimizer.optimize()`")
            raise ValueError("model not trained yet, `Optimizer.optimize()`")
        results = []
        for s_path, scenario in OptimizationOrchestrator._get_scenarios(path, scenario_filter=scenario_filter):
            scenario["eval_name"] = s_path.name
            scenario["general"]["num_period"] = num_period
            r = self.evaluate(scenario)
            r.index = pd.Index([s_path.name])
            results.append(r)
        return pd.concat(results, axis=0).sort_index()

    def evaluate(self, scenario: dict) -> pd.DataFrame:
        return self.model.evaluate(scenario)

    @staticmethod
    def evaluate_defaults_scenarios(
        num_period: int = 2000, path: str = "scenarios", scenario_filter: str = "phase"
    ) -> pd.DataFrame:
        """Evaluate the default scenario.

        Args:
            num_period (int, optional): number of simulation loop. Defaults to 2000.

        Returns:
            pd.DataFrame: MultiIndex DataFrame with results of the evaluation.
        """
        results = []
        for s_path, scenario in OptimizationOrchestrator._get_scenarios(path):
            scenario["eval_name"] = s_path.name
            scenario["general"]["num_period"] = num_period
            r = evaluate_single_scenario(scenario)
            r.index = pd.Index([s_path.name])
            results.append(r)
        return pd.concat(results).sort_index()

    def evaluate_base_scenario(self, config: dict | None = None) -> pd.DataFrame:
        if config is None:
            config = self.model.base_config
        return evaluate_single_scenario(config)

    def cprofile(self, init: bool = False, close: bool = False, result: bool = False, n: int = 10) -> None:
        """init or close cProfile for the optimization process. Can also access the results.

        Args:
            init (bool, optional): init cprofiler . Defaults to False.
            close (bool, optional): close cprofiler . Defaults to False.
            result (bool, optional): get cprofil result. Defaults to False.
            n (int, optional): n first function calls. Defaults to 10.

        """
        if init:
            if not self.enable_cprofile:
                self.profiler = NoProfiler()
            elif not getattr(self, "profiler"): # Create profiler if not existing
                self.log.info(Fore.YELLOW + f"=== Profiling enabled for optimization ===" + Fore.RESET)
                self.profiler = cProfile.Profile()
                self.profiler.enable()
            return

        if close and self.enable_cprofile:
            self.profiler.disable()
            self.log.info(Fore.YELLOW + "=== Profiling disabled ===" + Fore.RESET)
            return
        elif not self.enable_cprofile:
            return

        if init + close + result == 0:
            result = True  # Default to result if nothing is specified
        if result and self.enable_cprofile:
            stats = pstats.Stats(self.profiler, stream=sys.stdout)
            stats.sort_stats("cumulative")
            stats.print_stats(n)
        elif not self.enable_cprofile:
            self.log.info(Fore.YELLOW + "=== Profiling not enabled, no results to show ===" + Fore.RESET)

    def _start_model_solve(self, *args, **kwargs) -> None:
        self.cprofile(init=True)
        try:
            t = time.perf_counter()
            self.model.solve(*args, **kwargs)
            self.elapsed_time = time.perf_counter() - t
            time.sleep(0.5)  # Allow time for the solver to finish logging
        except Exception as e:
            self.cprofile(close=True)
            raise e
        
        if not kwargs.get("keep_alive", False):
            self.cprofile(close=True)

    def optimize(self, *args, **kwargs) -> None:
        """
        Run the optimization algorithm.
        """
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = self.verbose


        self._start_model_solve(*args, **kwargs)
        if self.elapsed_time:
            self.log.info(
                Fore.GREEN + f"=== Optimization completed in {self.elapsed_time:.2f} seconds ===" + Fore.RESET
            )
        self._save_history_in_cache(kwargs.get("model_cache", "last"))

    def optimize_across_phases(
        self, num_period: int = 2_000, log_score: bool = False, print_diffs: bool = False, save: bool = False, *args, **kwargs
    ) -> None:
        """Optimize the model across different phases.

        Args:
            num_period (int, optional): Number of periods for the optimization. Defaults to 2_000.
            log_score (bool, optional): Whether to log the score. Defaults to False.
            print_diffs (bool, optional): Whether to print differences. Defaults to False.
        """
        phases = {}
        for path, base_config in OptimizationOrchestrator._get_scenarios("scenarios", scenario_filter="phase"):
            phase = path.parts[1]
            if phase in phases.keys():
                continue
            base_config["general"]["num_period"] = num_period
            phases[phase] = base_config

        for i, phase in enumerate(sorted(phases.keys())):
            self.log.info(Fore.YELLOW + f"=== Starting optimization for phase: {phase} ===" + Fore.RESET)
            self.model.reset(phases[phase])

            keep_alive = True if i < len(phases) - 1 else False
            self.optimize(model_cache=phase, keep_alive=keep_alive, *args, **kwargs)
            if log_score:
                self.log_score()
            if print_diffs:
                self.compare_solution_to_base_config()
            if save:
                self.save_model(main_dir=f"./saved", save_dir=f"{phase}", save_name=f"_{phase}")

            self.log.info(Fore.YELLOW + f"=== Finished optimization for phase: {phase} ===\n" + Fore.RESET)

    def _save_history_in_cache(self, model_cache: str = "last") -> None:
        self.histories[model_cache] = {
            "scores": self.model.scores,
            "solution": self.model.solutions,
            "best_score": self.model.best_score,
            "best_solution": self.model.best_solution,
            "kpis": self.get_kpis(),
            "res": self.model.res if hasattr(self.model, "res") else None,
        }

    @property
    def scores_per_phases(self) -> dict:
        if self.histories == {}:
            raise ValueError("No scores computed, empty history.")
        return {phase: history["best_score"]["score"] for phase, history in self.histories.items()}

    def get_kpis(self) -> Kpis:
        if self.model.istrain is False:
            raise ValueError("model not trained yet, `Optimizer.optimize()`")

        sim, cfg = self.model.get_best_simulation()
        kpis = Kpis(sim.result, cfg)
        return kpis

    def plots_kpis(self, kpis: Kpis | int | None = None) -> None:
        """Plot the KPIs.
        Defaults to the current KPIs model if none are provided.

        Args:
            kpis (Kpis | int | None, optional): The KPIs to plot. if int use phase history. Defaults to None.
        """
        if kpis is None:
            kpis = self.get_kpis()
        elif isinstance(kpis, int):
            # kpis = [history['kpis'] for i, history in enumerate(self.histories.values()) if i == kpis][-1]
            kpis = list(self.histories.values())[kpis]["kpis"]

        for plot in kpis.generate_kpis_graphs():
            plot.show()

    def plot_model_performance(self, res: Result | int = None) -> None:
        # 1. Vérification des résultats
        if res is None:
            res = self.model.res
        elif isinstance(res, int):
            res = list(self.histories.values())[res]["res"]

        # 2. analyse approfondie
        analyzer = NSGA3HistoryAnalyzer(res)

        # Analyse de convergence
        metrics = analyzer.analyze_convergence()
        print("\n=== ANALYSE DE CONVERGENCE ===")
        print(f"Hypervolume final: {metrics['hypervolume'].iloc[-1]:.4f}")
        print(f"Amélioration totale: {metrics['hypervolume'].iloc[-1] - metrics['hypervolume'].iloc[0]:.4f}")
        print(f"Solutions finales: {metrics['n_solutions'].iloc[-1]}")

        # Détection de stagnation
        stagnation = analyzer.detect_stagnation(window_size=10, threshold=0.005)
        if stagnation["stagnation_ratio"] > 0.3:
            print("\n ATTENTION: Plus de 30% du temps en stagnation")

        # 4. visualisation complète
        fig1 = analyzer.plot_convergence()
        fig2 = analyzer.plot_stagnation_analysis()
        fig3 = analyzer.visualize_evolution()
        plt.show()

    def plot_pareto(self, scores: pd.DataFrame | int = None, figsize: tuple | list = (12, 12)) -> None:
        """Plot the pareto front of the optimization.

        Args:
            results (pd.DataFrame): DataFrame containing the results of the optimization. default use model best score

        return None: Displays the plots of the results.
        4x3 scatter plots of the results metrics.
        Each row corresponds to a metric, and each column corresponds to the other metrics.

        equivalent of :
        from pymoo.visualization.scatter import Scatter
        plot = Scatter()
        plot.add(model.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(model.res.F, facecolor="none", edgecolor="red")
        plot.show()

        """
        import matplotlib.pyplot as plt

        if scores is None:
            scores = self.model.scores
        elif isinstance(scores, int):
            scores = list(self.histories.values())[scores]["scores"]


        cost = scores["cost"]
        wasted_production = scores["wasted_production_over_time"]
        waiting_time = scores["waiting_time"]
        underfill_rate = scores["underfill_rate"]

        fig, axs = plt.subplots(4, 3, figsize=figsize)
        axs = axs.flatten()

        metrics_name = [
            "cost",
            "wasted_production_over_time",
            "waiting_time",
            "underfill_rate",
        ]
        metrics_values = [cost, wasted_production, waiting_time, underfill_rate]
        # Find best solution index
        best_idx = scores["score"].idxmin()
        
        for i, name in enumerate(metrics_name):
            ligne_i = 3 * i
            other_metrics = metrics_values[:i] + metrics_values[i + 1 :]
            other_metrics_name = metrics_name[:i] + metrics_name[i + 1 :]
            
            # Plot all points in blue
            axs[ligne_i].scatter(metrics_values[i], other_metrics[0], c='blue', alpha=0.6)
            axs[ligne_i + 1].scatter(metrics_values[i], other_metrics[1], c='blue', alpha=0.6)
            axs[ligne_i + 2].scatter(metrics_values[i], other_metrics[2], c='blue', alpha=0.6)
            
            # Highlight best solution in red
            kw = dict(c='red', s=100, alpha=0.8, edgecolor='darkred', linewidth=2)
            axs[ligne_i].scatter(metrics_values[i][best_idx], other_metrics[0][best_idx], **kw)
            axs[ligne_i + 1].scatter(metrics_values[i][best_idx], other_metrics[1][best_idx], **kw)
            axs[ligne_i + 2].scatter(metrics_values[i][best_idx], other_metrics[2][best_idx], **kw)

            for j, ax in enumerate(axs[ligne_i : ligne_i + 3]):
                ax.set_xlabel(name)
                ax.set_ylabel(other_metrics_name[j])
                fig.tight_layout()
        plt.show()

    def log_score(self) -> None:
        """
        Log the score of the current best solution.
        """
        self.model.log_score()

    def save_model(
        self, main_dir: str = "./saved/", save_dir: str = "model_files", save_name: str = "", index: bool = True
    ) -> None:
        if not self.model.istrain:
            self.log.info("model not trained yet, `self.solve()`")
            raise ValueError("model not trained yet, `self.solve()`")
        main_dir = Path(main_dir) / save_dir
        main_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self.model.data_to_saved().items():
            if "model" in name or "results" in name:
                with open(main_dir / f"{name + save_name}.dill", "wb") as f:
                    dill.dump(data, f)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(main_dir / f"{name + save_name}.csv", index=index)
            elif isinstance(data, np.ndarray):
                np.save(main_dir / f"{name + save_name}.npy", data)
            else:
                self.log.warning(Fore.YELLOW + f"Skipping saving {name} as it is not a DataFrame." + Fore.RESET)
        self.log.info(Fore.GREEN + "=== Resultats Sauvegarde ===" + Fore.RESET)
        self.log.info(f"Result files saved in {Fore.CYAN + str(main_dir.resolve()) + Fore.RESET} directory")

    @classmethod
    def from_phases(cls, model_name: str, phases_dir: str | Path, base_config: dict, logger: Logger | None = None, **kwargs) -> None:
        """Create an OptimizationOrchestrator instance from multiple phases.

        Args:
            phases_dir: Directory containing phase configurations.
            logger: Optional logger instance.
            **kwargs: Additional arguments passed to the OptimizationOrchestrator constructor.
        """
        log = logger or Logger()
        paths = []
        if isinstance(phases_dir, str):
            phases_dir = Path(phases_dir)
        for path in phases_dir.glob("*"):
            if path.is_dir() and "phase" in path.name:
                paths.append(path)
                
        if not paths:
            raise ValueError(f"No valid phase directories found in {phases_dir}")
        
        instance = cls(model=None, logger=log, **kwargs) 
        paths = sorted(paths, key=lambda x: x.name)
        for p in paths:
            log.info(Fore.CYAN + f"Found phase directory: {p}" + Fore.RESET)
            model = instance._load_model(model_name, sol_dir_path=p, logger=log, base_config=base_config, **kwargs)
            instance.model = model
            instance._save_history_in_cache(model_cache=p.name)

        return instance

    @classmethod
    def from_model(cls, model_name:str, sol_dir_path:str|Path, base_config: dict, logger: Logger | None = None, **kwargs) -> None:
        """Create an OptimizationOrchestrator instance from a model.

        Args:
            model: The optimization model instance.
            logger: Optional logger instance.
            **kwargs: Additional arguments passed to the OptimizationOrchestrator constructor.
        """
        log = logger or Logger()
        model = cls._load_model(model_name=model_name, sol_dir_path=sol_dir_path, logger=log, base_config=base_config, **kwargs)
        return cls(model=model, logger=log, **kwargs)
    
    @staticmethod
    def _load_model(model_name:str, sol_dir_path: str | Path, base_config: dict, logger: Logger | None = None, **kwargs):
        """Load a pre-trained model from saved solutions.

        Args:
            model: The model type to load (e.g., "GaModel", "CpModel")
            sol_dir_path: Path to directory containing the saved CSV files
            base_config: Base configuration (optional, uses current if not provided)
            **kwargs: Additional arguments passed to the model's load method
        """

        # Determine model type and load accordingly
        if model_name == "GaModel":  # GaModel
            model = GaModel.load(sol_dir_path=sol_dir_path, base_config=base_config, logger=logger, **kwargs)    
        elif model_name == "CpModel":  # CpModel
            model = CpModel.load(sol_dir_path=sol_dir_path, base_config=base_config, logger=logger, **kwargs)
        else:
            raise ValueError("Unknown model type for loading")

        return model

    def build_config_from_solution(self, solution: dict, algorithm: str | None = None, *args, **kwargs) -> dict:
        """
        Build a configuration dictionary from a solution.

        Args:
            solution (dict): The solution dictionary to build the configuration from.
            from_model (bool): If True, use the model's method to get the configuration.

        Returns:
            dict: The built configuration dictionary.
        """
        return self.model.cfg_builder.get_config_from_solution(
            solution, algorithm=algorithm or self.model.algorithm_name, *args, **kwargs
        )

    def render_best_solution(self, *args, **kwargs) -> None:
        config = self.build_config_from_solution(self.model.best_solution, *args, **kwargs)
        self._run_animation(config)

    def render_solution(self, solution=None, *args, **kwargs) -> None:
        if solution is None:
            solution = self.model.best_solution
        elif isinstance(solution, int):
            solution = list(self.histories.values())[solution]["best_solution"]


        config = self.build_config_from_solution(solution, *args, **kwargs)
        self._run_animation(config)

    def render_heuristic_solution(self, solution: dict) -> None:
        config = self.build_config_from_solution(solution, mode="heuristic")
        self._run_animation(config)

    def _run_animation(self, config: dict) -> None:
        from GUI import PGAnime

        print(config)
        PGAnime(config).run()


if __name__ == "__main__":
    import argparse
    from optimizer.CPModel.cp_model import CpModel
    from optimizer.GAModel.ga_model import GaModel
    from eco2_normandy.tools import get_simlulation_variable

    parser = argparse.ArgumentParser(
        description="Solve the scheduling model with a configurable number of CP-SAT iterations."
    )
    parser.add_argument(
        "-t",
        "--max_time",
        type=int,
        default=60 * 60,  # default 1 hour
        help="Maximum time in seconds for the CP-SAT solver to run.",
    )
    parser.add_argument(
        "-s",
        "--saved-folder",
        type=str,
        default="",
        help="Folder to save the results and solutions, within './saved/' folder.",
    )
    parser.add_argument(
        "-m",
        "--max-eval",
        type=int,
        default=5,
        help="Maximum number of evaluations (with valid result) for the CP-SAT callback",
    )
    args = parser.parse_args()
    saved_folder = Path("./saved/") / args.saved_folder

    path = "scenarios/dev/phase3_bergen_18k_2boats.yaml"
    config = get_simlulation_variable(path)[0]
    config["general"]["num_period"] = 2_000

    logger = Logger()
    model = GaModel(config, pop_size=100, n_gen=10, parallelization=True, algorithm_name="NSGA3")
    # model = CpModel(config)
    optimizer = OptimizationOrchestrator(model=model, verbose=1, enable_cprofile=False)
    # optimizer.optimize(max_evals=5, verbose=1, max_time_in_seconds=1000)
    optimizer.optimize()
    optimizer.log_score()
    optimizer.save_model(main_dir=str(saved_folder))

    yesno = input("Do you want to visualize the best simulation? (y/n): ").strip().lower()
    if yesno == "y":
        optimizer.render_best_solution()
    else:
        print("Simulation visualization skipped.")
