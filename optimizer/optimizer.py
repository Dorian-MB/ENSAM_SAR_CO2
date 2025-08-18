import time
import cProfile, pstats
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from colorama import Fore

from eco2_normandy.logger import Logger
from optimizer.utils import get_all_scenarios, ConfigBuilderFromSolution, NoProfiler, evaluate_single_scenario
from optimizer.compare_scenarios import print_diffs

metrics_keys = ["cost", "wasted_production_over_time", "waiting_time", "underfill_rate"]
metrics_weight = [20, 20, 15, 30]
class Optimizer:
    """
    Base class for optimization algorithms.
    This class provides a framework for implementing various optimization strategies.
    """

    def __init__(self, model, logger:Logger=None, verbose:int|bool=1,
                metrics_keys=metrics_keys, metrics_weight=metrics_weight,
                enable_cprofile=False)->None:
        self.log = logger or Logger()
        self.model = model
        self.verbose = verbose
        self.metrics_keys = metrics_keys
        self.metrics_weight = metrics_weight
        self.enable_cprofile = enable_cprofile
        self.profiler = None
        self.cfg_builder = ConfigBuilderFromSolution(model.base_config)
        self.full_results = None

    def set_model_callback(self, *args, **kwargs)->None:
        """
        Set the callback for the optimizer.
        """
        self.model._set_callback(*args, **kwargs)
    
    def set_base_config(self, base_config:dict)->None:
        """
        Set the base configuration for the optimizer.
        """
        self.model.base_config = base_config
        self.model.callbacks.base_config = base_config # todo : erreur si GaModel
        self.cfg_builder = ConfigBuilderFromSolution(base_config)
    
    def compare_solution_to_base_config(self, solution):
        sol_cfg = self.cfg_builder.build(solution)
        print_diffs(sol_cfg, self.model.base_config) # todo: ajouter un print pour mieux differencier les 2 scenarios
    
    def evaluate_all_scenarios(self, num_period:int=2000, 
                        path:str="scenarios/", 
                        scenario_filter:str="phase",
                )->pd.DataFrame:
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
            return
        results = []
        for s_path, scenario in get_all_scenarios(path):
            if scenario_filter and scenario_filter not in str(s_path.parent):
                continue
            scenario['eval_name'] = s_path.name
            scenario["general"]["num_period"] = num_period
            r = self.evaluate(scenario)
            r.index = pd.Index([s_path.name])
            results.append(r)
        return pd.concat(results, axis=0).sort_index()

    def evaluate(self, scenario):
        return self.model.evaluate(scenario)

    @staticmethod
    def evaluate_defaults_scenarios(num_period=2000, path:str="scenarios/")->pd.DataFrame:
        """Evaluate the default scenario.

        Args:
            num_period (int, optional): number of simulation loop. Defaults to 2000.

        Returns:
            pd.DataFrame: MultiIndex DataFrame with results of the evaluation.
        """
        results = []
        for s_path, scenario in get_all_scenarios(path):
            scenario['eval_name'] = s_path.name
            scenario["general"]["num_period"] = num_period
            r = evaluate_single_scenario(scenario)
            r.index = pd.Index([s_path.name])
            results.append(r)
        return pd.concat(results).sort_index()

    def evaluate_base_scenario(self):
        return evaluate_single_scenario(self.model.base_config)

    def plot_pareto(self, scores:pd.DataFrame=None, figsize:tuple=(15, 15))->None:
        """Plot the pareto front of the optimization.

        Args:
            results (pd.DataFrame): DataFrame containing the results of the optimization.

        return None: Displays the plots of the results.
        4x3 scatter plots of the results metrics.
        Each row corresponds to a metric, and each column corresponds to the other metrics.
        """
        import matplotlib.pyplot as plt
        scores = scores if scores is not None else self.model.scores

        cost = scores['cost']
        wasted_production = scores['wasted_production_over_time']
        waiting_time = scores['waiting_time']
        underfill_rate = scores['underfill_rate']

        fig, axs = plt.subplots(4, 3, figsize=figsize)
        axs = axs.flatten()

        metrics_name = ['cost', 'wasted_production_over_time', 'waiting_time', 'underfill_rate']
        metrics_values = [cost, wasted_production, waiting_time, underfill_rate]
        for i, name in enumerate(metrics_name):
            ligne_i = 3 * i
            other_metrics = metrics_values[:i] + metrics_values[i+1:]
            other_metrics_name = metrics_name[:i] + metrics_name[i+1:]
            axs[ligne_i].scatter(metrics_values[i], other_metrics[0])
            axs[ligne_i+1].scatter(metrics_values[i], other_metrics[1])
            axs[ligne_i+2].scatter(metrics_values[i], other_metrics[2])
            for j, ax in enumerate(axs[ligne_i:ligne_i+3]):
                ax.set_xlabel(name)
                ax.set_ylabel(other_metrics_name[j])
                fig.tight_layout()
        plt.show()
            
    def cprofile(self, profiler=None, init=False, close=False, result=False, n:int=10):
        """ init or close cProfile for the optimization process. Can also access the results.

        Args:
            profiler (_type_, optional): instance of cProfile.Profile(). Defaults to None.
            init (bool, optional): init cprofiler . Defaults to False.
            close (bool, optional): close cprofiler . Defaults to False.
            result (bool, optional): get cprofil result. Defaults to False.
            n (int, optional): n first function calls. Defaults to 10.

        Returns:
            _type_: None
        """
        assert init + close + result <= 1, "Only one of init, close, or result can be True."
        if init:
            if not self.enable_cprofile:
                return NoProfiler()
            self.log.info(Fore.YELLOW + f"=== Profiling enabled for optimization ===" + Fore.RESET)
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            return 
                
        profiler = profiler or self.profiler
        if close and self.enable_cprofile:
            profiler.disable()
            self.log.info(Fore.YELLOW + "=== Profiling disabled ===" + Fore.RESET)
            return
        elif not self.enable_cprofile:
            return
        
        
        if result and self.enable_cprofile:
            stats = pstats.Stats(profiler, stream=sys.stdout)
            stats.sort_stats('cumulative')
            stats.print_stats(n)
        elif not self.enable_cprofile:
            self.log.info(Fore.YELLOW + "=== Profiling not enabled, no results to show ===" + Fore.RESET)

    def _start_model_solve(self, *args, **kwargs):
        self.cprofile(init=True)
        try:
            t = time.perf_counter()
            self.model.solve(*args, **kwargs)
            self.elapsed_time = time.perf_counter() - t 
            time.sleep(0.5)  # Allow time for the solver to finish logging
        except Exception as e:
            self.log.error(f"Error occurred during model solve: {e}")
            raise e
        finally:
            self.cprofile(close=True)
        
    def optimize(self, *args, **kwargs):
        """
        Run the optimization algorithm.
        """
        if kwargs.get('verbose') is None:
            kwargs['verbose'] = self.verbose
        self._start_model_solve(*args, **kwargs)
        if self.elapsed_time: self.log.info(Fore.GREEN + f"=== Optimization completed in {self.elapsed_time:.2f} seconds ===" + Fore.RESET)
        
    def log_score(self):
        """
        Log the score of the current best solution.
        """
        self.model.log_score()
        
    def save_solution(self, dir:str='./saved/', index:bool=True):
        if not self.model.istrain:
            self.log.info("model not trained yet, `self.solve()`")
            return
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        for name, df in self.model.data_to_saved().items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(dir / f"{name}.csv", index=index)
            else:
                self.log.warning(Fore.YELLOW + f"Skipping saving {name} as it is not a DataFrame." + Fore.RESET)
        self.log.info(Fore.GREEN + "=== Résultats Sauvegardé ===" + Fore.RESET)
        self.log.info(f"Result files saved in {Fore.CYAN+str(dir.resolve())+Fore.RESET} directory")

    def build_config_from_solution(self, solution:dict, algorithm:str=None, *args, **kwargs) -> dict:
        """
        Build a configuration dictionary from a solution.
        
        Args:
            solution (dict): The solution dictionary to build the configuration from.
            from_model (bool): If True, use the model's method to get the configuration.
        
        Returns:
            dict: The built configuration dictionary.
        """
        return self.cfg_builder.get_config_from_solution(solution, algorithm=algorithm or self.model.algorithm_name,
                                                        *args, **kwargs)

    def render_best_solution(self, *args, **kwargs):
        config = self.build_config_from_solution(self.model.best_solution, *args, **kwargs)
        self._run_animation(config)

    def render_solution(self, solution, *args, **kwargs):
        config = self.build_config_from_solution(solution, *args, **kwargs)
        self._run_animation(config)

    def render_heuristic_solution(self, solution):
        config = self.build_config_from_solution(solution, mode="heuristic")
        self._run_animation(config)

    def _run_animation(self, config):
        from GUI import PGAnime
        print(config)
        PGAnime(config).run()
        
if __name__ == "__main__":

    import argparse
    from optimizer.cp_model import CpModel
    from optimizer.ga_model import GAModel
    from eco2_normandy.tools import get_simlulation_variable
    parser = argparse.ArgumentParser(
        description="Solve the scheduling model with a configurable number of CP-SAT iterations."
    )
    parser.add_argument(
        "-t", "--max_time",
        type=int,
        default=60 * 60,  # default 1 hour
        help="Maximum time in seconds for the CP-SAT solver to run."
    )
    parser.add_argument(
        "-s", "--saved-folder",
        type=str,
        default="",
        help="Folder to save the results and solutions, within './saved/' folder."
    )
    parser.add_argument(
        "-m", "--max-eval",
        type=int,
        default=5,
        help="Maximum number of evaluations (with valid result) for the CP-SAT callback"
    )
    args = parser.parse_args()
    saved_folder = Path("./saved/") / args.saved_folder

    path = "scenarios/dev/phase3_bergen_18k_2boats.yaml"
    config = get_simlulation_variable(path)[0]
    config["general"]["num_period"] = 2_000

    logger = Logger()
    model = GAModel(config, pop_size=100, n_gen=10, parallelization=True, algorithm="NSGA3")
    # model = CpModel(config)
    optimizer = Optimizer(model=model, verbose=1, enable_cprofile=False)
    # optimizer.optimize(max_evals=5, verbose=1, max_time_in_seconds=1000)
    optimizer.optimize()
    optimizer.log_score()
    optimizer.save_solution(dir=str(saved_folder))

    yesno = input("Do you want to visualize the best simulation? (y/n): ").strip().lower()
    if yesno == "y":
        optimizer.render_best_solution()
    else:
        print("Simulation visualization skipped.")

