import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import pandas as pd
from colorama import Fore

from eco2_normandy.logger import Logger
from optimizer.callback import SimCallback
from optimizer.utils import flatten, ConfigBuilderFromSolution, Normalizer
from optimizer.boundaries import ConfigBoundaries

from ortools.sat.python import cp_model

metrics_keys = Normalizer().metrics_keys
metrics_weight = Normalizer().metrics_weight


class CpModel(cp_model.CpModel):
    def __init__(
        self,
        config: dict,
        algorithm: str = "SearchForAllSolutions",
        caps_step: int = 1000,
        metrics_keys: list = metrics_keys,
        metrics_weight: list = metrics_weight,
        logger=None,
        verbose: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_config = config
        self.metrics_keys = metrics_keys
        self.metrics_weight = metrics_weight
        self.caps_step = caps_step
        self.verbose = verbose
        self.boundaries = ConfigBoundaries(verbose=0)
        self.log = logger or Logger()
        self.solver = cp_model.CpSolver()
        self.algorithm_name = algorithm
        self.Algoritm = self._get_alogorithm(algorithm)
        self.callback = None
        self.callback_vars = []
        self.surrogate_vars = []
        self.vars = {}
        self.istrain = False
        self.max_time_in_seconds = None
        self.cfg_builder = ConfigBuilderFromSolution(config, self.boundaries)

    def reset(self, base_config: dict):
        self.istrain = False
        self.base_config = base_config
        self.callback_vars = []
        self.surrogate_vars = []
        self.vars = {}
        self.cfg_builder = ConfigBuilderFromSolution(base_config, self.boundaries)

    def _get_alogorithm(self, algorithm):
        if algorithm == "SearchForAllSolutions":
            return self.SearchForAllSolutions
        elif algorithm == "HybridSearch":
            return self.HybridSearch
        elif algorithm == "HeuristicSolve":
            return self.HeuristicSolve
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. Available algorithms: 'SearchForAllSolutions', 'HybridSearch', 'HeuristicSolve'."
            )

    def _add_int_variables(self, min: int | list, max: int | list, names: list = [], name: str = ""):
        """Helper function to add integer variables with a name."""
        if name not in self.vars:
            if isinstance(min, list) and isinstance(max, list) and isinstance(names, list):
                self.vars[name] = [self.NewIntVar(int(mn), int(mx), nm) for mn, mx, nm in zip(min, max, names)]
            else:
                self.vars[name] = self.NewIntVar(int(min), int(max), name)

    def _add_callback_objective(self):
        # Storages
        max_num_storages = self.boundaries.max_num_storages
        # Ships
        ship_cap_min, ship_cap_max = (
            self.boundaries.ship_capacity["min"],
            self.boundaries.ship_capacity["max"],
        )
        max_num_ships = self.boundaries.max_num_ships
        min_speed, max_speed = (
            self.boundaries.ship_speed["min"],
            self.boundaries.ship_speed["max"],
        )
        storage_caps_min, storage_caps_max = (
            self.boundaries.storage_caps["min"],
            self.boundaries.storage_caps["max"],
        )
        tanks_min, tanks_max = (
            self.boundaries.factory_tanks["min"],
            self.boundaries.factory_tanks["max"],
        )
        initial_destination = self.boundaries.initial_destination  # 0=factory, 1=Rotterdam, 2=Bergen
        fixed_storage_destination = self.boundaries.fixed_storage_destination  # 1=Bergen, 0=Rotterdam

        #### Variables CP-SAT ####
        #! ---- Variables pour les storages ----
        self._add_int_variables(1, max_num_storages, name="num_storages")

        #! ---- Variables pour les navires ----
        self._add_int_variables(1, max_num_ships, name="num_ship")

        self._add_int_variables(
            min=[0] * max_num_ships,
            max=[initial_destination] * max_num_ships,
            names=[f"init{i + 1}_destination" for i in range(max_num_ships)],
            name="initial_ship_destination",
        )

        self._add_int_variables(
            min=[0] * max_num_ships,
            max=[fixed_storage_destination] * max_num_ships,
            names=[f"fixed{i + 1}_storage_destination" for i in range(max_num_ships)],
            name="fixed_storage_destination",
        )

        self._add_int_variables(ship_cap_min, ship_cap_max, name="ship_capacity")
        self._add_int_variables(min_speed, max_speed, name="ship_speed")

        self.vars["ship_used"] = [self.NewBoolVar(f"ship{i + 1}_used") for i in range(max_num_ships)]
        self.Add(sum(self.vars["ship_used"]) == self.vars["num_ship"])

        for i in range(max_num_ships):
            self.Add(self.vars["initial_ship_destination"][i] == 0).OnlyEnforceIf(self.vars["ship_used"][i].Not())
            self.Add(self.vars["fixed_storage_destination"][i] == 0).OnlyEnforceIf(self.vars["ship_used"][i].Not())

        self._add_int_variables(
            ship_cap_min // self.caps_step,
            ship_cap_max // self.caps_step,
            name="ship_units",
        )
        self.Add(self.vars["ship_capacity"] == self.caps_step * self.vars["ship_units"])

        ship_used = self.vars["ship_used"]
        [
            self.Add(ship_used[i] >= ship_used[i + 1]) for i in range(max_num_ships - 1)
        ]  # les navires utilisés sont consécutifs

        self.vars["use_Bergen"] = self.NewBoolVar("use_Bergen")
        self.vars["use_Rotterdam"] = self.NewBoolVar("use_Rotterdam")
        self.Add(self.vars["use_Bergen"] + self.vars["use_Rotterdam"] == self.vars["num_storages"])

        for i in range(max_num_ships):
            self.Add(self.vars["initial_ship_destination"][i] != 2).OnlyEnforceIf(self.vars["use_Bergen"].Not())
            self.Add(self.vars["fixed_storage_destination"][i] != 1).OnlyEnforceIf(self.vars["use_Bergen"].Not())

            self.Add(self.vars["initial_ship_destination"][i] != 1).OnlyEnforceIf(self.vars["use_Rotterdam"].Not())
            self.Add(self.vars["fixed_storage_destination"][i] != 0).OnlyEnforceIf(self.vars["use_Rotterdam"].Not())

        self._add_int_variables(min=storage_caps_min, max=storage_caps_max, name="storage_caps")
        self._add_int_variables(
            storage_caps_min // self.caps_step,
            storage_caps_max // self.caps_step,
            name="storage_units",
        )
        self.Add(self.vars["storage_caps"] == self.caps_step * self.vars["storage_units"])
        self._add_int_variables(min=tanks_min, max=tanks_max, name="number_of_tanks")

        self.callback_vars = flatten(self.vars.values())

    def _add_heuristic_objectives(self):
        ship_cap_min, ship_cap_max = (
            self.boundaries.ship_capacity["min"],
            self.boundaries.ship_capacity["max"],
        )
        max_ships = self.boundaries.max_num_ships
        min_speed, max_speed = (
            self.boundaries.ship_speed["min"],
            self.boundaries.ship_speed["max"],
        )

        # Decision vars subset for heuristic phase
        # Share with callback phase
        self._add_int_variables(1, max_ships, name="num_ship")
        self._add_int_variables(ship_cap_min, ship_cap_max, name="ship_capacity")
        self._add_int_variables(min_speed, max_speed, name="ship_speed")

        num_ship = self.vars["num_ship"]
        ship_speed = self.vars["ship_speed"]
        ship_caps = self.vars["ship_capacity"]

        # Constants from config/boundaries
        total_period = self.base_config["general"]["num_period"]
        cost_per_ship = self.base_config["ships"][0]["ship_buying_cost"]
        fuel_price = self.base_config["KPIS"]["fuel_price_per_ton"]
        sources = self.base_config["factory"]["sources"]
        sources_annual_prod = [src["annual_production_capacity"] for src in sources]
        prod_per_year = sum(sources_annual_prod)
        prod_rate = prod_per_year / (24 * 365 * self.base_config["general"]["num_period_per_hours"])

        # Normalization & weights
        n_wasted = max(1, int(prod_rate * total_period))  # max wasted
        n_cost = max(1, int(cost_per_ship * max_ships + fuel_price * max_speed * max_ships))
        w = {k: v for k, v in zip(self.metrics_keys, self.metrics_weight)}

        # Transport capacity: ship_caps * num_ship
        max_cap = ship_cap_max
        transport_cap_max = max_cap * max_ships
        transport_cap = self.NewIntVar(0, transport_cap_max, "transport_cap")
        self.AddMultiplicationEquality(transport_cap, [ship_caps, num_ship])

        # wasted = max(0, prod_max - transport_cap)
        prod_max = int(prod_rate * total_period)
        wasted = self.NewIntVar(0, prod_max, "wasted_slack")
        minus_t = self.NewIntVar(-transport_cap_max, prod_max, "minus_transport")
        self.Add(minus_t == prod_max - transport_cap)
        zero = self.NewIntVar(0, 0, "zero")
        self.AddMaxEquality(wasted, [zero, minus_t])

        # surrogate cost proxy: investment + operating proxy
        total_ship_speed_time = self.NewIntVar(0, max_speed * max_ships, "total_ship_speed_time")
        self.AddMultiplicationEquality(total_ship_speed_time, [ship_speed, num_ship])
        cost_expr = cost_per_ship * num_ship + fuel_price * total_ship_speed_time

        # Inverse speed approximation via a table constraint inv ≈ floor(scale / speed) ~ time = d/v
        scale = 60
        proxy_time = self.NewIntVar(0, max(1, scale // min_speed), "proxy_time")
        # Build allowed pairs for (ship_speed, proxy_time)
        pairs = []
        for s in range(min_speed, max_speed + 1):
            inv = scale // s
            pairs.append([s, inv])
        self.AddAllowedAssignments([ship_speed, proxy_time], pairs)

        # waiting proxy ~ num_ship * inv_speed
        waiting = self.NewIntVar(0, max_ships * max(1, scale // min_speed), "waiting_proxy")
        self.AddMultiplicationEquality(waiting, [num_ship, proxy_time])

        surrogate = (
            w["cost"] / n_cost * cost_expr
            + w["wasted_production_over_time"] / n_wasted * wasted
            + w["waiting_time"] / max(1, total_period) * waiting
        )
        self.Minimize(surrogate)
        self.surrogate = surrogate
        self.surrogate_vars = [ship_caps, ship_speed, num_ship]

    def _set_callback(
        self,
        Callback=SimCallback,
        max_evals=50,
        verbose=None,
        max_time_in_seconds=None,
        metrics_keys=None,
    ):
        self.callback = Callback(
            variables=self.callback_vars,
            base_config=self.base_config,
            max_evals=max_evals,
            metrics_keys=metrics_keys or self.metrics_keys,
            boundaries=self.boundaries,
            verbose=verbose or self.verbose,
        )
        if isinstance(max_time_in_seconds, (int, float)):
            self.solver.parameters.max_time_in_seconds = int(max_time_in_seconds)

    def HeuristicSolve(self, max_time_in_seconds: int = 30) -> dict:
        """Phase 1: CP‐only solve with surrogate objective."""
        self._add_heuristic_objectives()

        # set a short time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = int(max_time_in_seconds)

        # (optional) you can hint from a previous solve here

        status = solver.Solve(self)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("Heuristic solve failed to find any solution")
        # collect your solution
        self.heuristic_sol = {v.Name(): solver.Value(v) for v in self.surrogate_vars}
        if self.verbose >= 1:
            self.log.info(Fore.BLUE + "=== Heuristic solution ===" + Fore.RESET)
            self.log.info(self.heuristic_sol)
        return self.heuristic_sol

    def HybridSearch(
        self,
        max_time_callback: int | None = None,
        max_time_heuristic: int = 30,
        max_time_in_seconds: int | None = None,
        **kwargs,
    ) -> None:
        if max_time_callback is None:
            max_time_callback = max_time_in_seconds
        # 1) Heuristic phase
        if self.verbose:
            self.log.info(Fore.GREEN + "=== Phase 1: Heuristic solve ===" + Fore.RESET)
        sol = self.HeuristicSolve(max_time_in_seconds=max_time_heuristic)
        self.ClearObjective()

        for var in self.surrogate_vars:
            self.AddHint(var, sol[var.Name()])

        # 3) Full enumeration with Callback
        if self.verbose:
            self.log.info(Fore.GREEN + "=== Phase 2: Full enumeration with SimCallback ===" + Fore.RESET)
        self.SearchForAllSolutions(max_time_in_seconds=max_time_callback, **kwargs)

    def SearchForAllSolutions(self, Callback=SimCallback, **kwargs) -> None:
        self.istrain = True
        self._add_callback_objective()
        self._set_callback(Callback=Callback, **kwargs)
        self.solver.parameters.enumerate_all_solutions = True
        self.solver.solve(self, self.callback)
        self.callback.set_results()

    def solve(self, *args, **kwargs):
        try:
            self.Algoritm(*args, **kwargs)
        except Exception as e:
            self.log.error(Fore.RED + f"An error occurred while running `{self.algorithm_name}`" + Fore.RESET)
            raise e

    def get_best_simulation(self, num_period: int = 2000) -> tuple[dict, dict]:
        if self.istrain is False:
            raise RuntimeError("No results available. Call solve() first.")
        best_cfg = self.cfg_builder.build(self.best_solution, num_period=num_period)
        sim = self.callback.run_simulation(best_cfg)
        return sim, best_cfg

    def log_score(self) -> None:
        best_score = self.best_score
        if hasattr(self, "heuristic_sol"):
            score_heuristic = self.evaluate(self.cfg_builder.build_heuristic(self.heuristic_sol)).iloc[0]
            self.log.info(Fore.GREEN + "=== Heuristic solution ===" + Fore.RESET)
            self.log.info(f"\n{score_heuristic}")
            self.log.info(Fore.BLUE + f"Score heuristic: {score_heuristic['score']:,.0f}" + Fore.RESET)
        else:
            self.log.info(Fore.LIGHTRED_EX + "=== No heuristic solution found ===" + Fore.RESET)
        if best_score is not None:
            self.log.info(Fore.GREEN + "=== Résultats des évaluations avec callback ===" + Fore.RESET)
            self.log.info(
                Fore.LIGHTBLUE_EX
                + f"Best score : {best_score['score']:,.0f}, total solution tested: {self.callback.solutions_tested}"
                + Fore.RESET
            )
            self.log.info(Fore.BLUE + f"=>coût={best_score['cost']:,.0f} €:" + Fore.RESET)
            self.log.info(Fore.BLUE + f"=>perte={best_score['wasted_production_over_time']:,.0f} m^3:" + Fore.RESET)
            self.log.info(Fore.BLUE + f"=>wating_time={best_score['waiting_time']:,.0f}s :" + Fore.RESET)
            self.log.info(Fore.BLUE + f"=>underfill_rate={best_score['underfill_rate'] * 100:,.2f}% :\n" + Fore.RESET)
        else:
            self.log.info(Fore.LIGHTRED_EX + "=== Aucune callback solution trouvée ===" + Fore.RESET)

    @property
    def scores(self) -> pd.DataFrame:
        return self.callback.raw_metrics

    @property
    def solutions(self) -> pd.DataFrame:
        return self.callback.solutions

    @property
    def best_score(self) -> pd.Series:
        return self.callback.best_raw_score()

    @property
    def best_solution(self) -> pd.Series:
        return self.solutions.loc[self.best_score.name]

    @property
    def pareto_front(self) -> pd.DataFrame:
        return pd.DataFrame(self.callback.pareto_front.get_front()).T

    def data_to_saved(self) -> dict:
        name = "CP"
        return {
            f"solutions_{name}": self.solutions,
            f"scores_{name}": self.scores,
            f"pareto_{name}": self.pareto_front,
        }

    def evaluate(self, cfg: dict, clip: bool = True) -> pd.DataFrame:
        """
        Evaluate the configuration.
        This methode is meant to be used after the model has been trained.
        """
        if self.istrain is False:
            self.log.error(
                Fore.RED
                + "Model not trained yet. Please train the model before evaluating a configuration."
                + Fore.RESET
            )
            raise ValueError("Model not trained yet.")
        cfg = ConfigBuilderFromSolution(cfg, self.boundaries).build(self.best_solution)
        sim = self.callback.run_simulation(cfg)
        if sim is None:
            self.log.error(
                Fore.RED
                + f"Simulation {cfg.get('eval_name', '')} failed. Please check the configuration and try again."
                + Fore.RESET
            )
            raise ValueError("Simulation failed.")
        metrics = self.callback.calculate_performance_metrics(cfg, sim)
        norm_metrics = self.callback.normalize(metrics, clip=clip)
        metrics["score"] = self.callback.normalize.compute_score(norm_metrics)
        return metrics

    @classmethod
    def load(
        cls,
        sol_dir_path: str | Path,
        base_config: dict,
        algorithm: str = "SearchForAllSolutions",
        caps_step: int = 1000,
        metrics_keys: list = metrics_keys,
        metrics_weight: list = metrics_weight,
        logger=None,
        verbose: int = 1,
        **kwargs,
    ) -> "CpModel":
        """Load solutions from a CSV file"""
        log = logger or Logger()
        if isinstance(sol_dir_path, str):
            sol_dir_path = Path(sol_dir_path)
        if not sol_dir_path.exists():
            log.error(Fore.RED + f"Solution {sol_dir_path} not found." + Fore.RESET)
            raise FileNotFoundError(f"Solution {sol_dir_path} not found.")
        if not sol_dir_path.is_dir():
            raise NotADirectoryError(f"Solution path {sol_dir_path} is not a directory.")

        csv_files = list(sol_dir_path.glob("*.csv"))
        if not csv_files or len(csv_files) != 3:
            log.error(
                Fore.RED
                + f"No CSV files found in {sol_dir_path}. Needed 3 CSV files, got {len(csv_files)}"
                + Fore.RESET
            )
            raise FileNotFoundError(f"No CSV files found in {sol_dir_path}.")

        # Load CSV files
        solutions_df = None
        scores_df = None
        pareto_df = None

        for csv in csv_files:
            df = pd.read_csv(str(csv), index_col=0)
            if "solution" in csv.name:
                solutions_df = df
                solutions_df.index.name = "solution_id"
            elif "score" in csv.name:
                scores_df = df
                scores_df.index.name = "solution_id"
            elif "pareto" in csv.name:
                pareto_df = df
                pareto_df.index.name = "solution_id"
            else:
                log.error(Fore.RED + f"Unknown CSV file: {csv.name}" + Fore.RESET)
                raise ValueError(f"Unknown CSV file: {csv.name}")

        # Verify all files were loaded
        if solutions_df is None or scores_df is None or pareto_df is None:
            missing = []
            if solutions_df is None:
                missing.append("solutions")
            if scores_df is None:
                missing.append("scores")
            if pareto_df is None:
                missing.append("pareto")
            raise ValueError(f"Missing required CSV files: {missing}")

        log.info(Fore.GREEN + f"Loading solutions from {Fore.LIGHTCYAN_EX}{str(sol_dir_path)}" + Fore.RESET)

        # Create instance with loaded data
        instance = cls(
            config=base_config,
            algorithm=algorithm,
            caps_step=caps_step,
            metrics_keys=metrics_keys,
            metrics_weight=metrics_weight,
            logger=log,
            verbose=verbose,
            **kwargs,
        )

        # Create a mock callback with loaded data
        from optimizer.callback import SimCallback

        instance.callback = SimCallback(
            variables=[],  # Empty for loaded model
            base_config=base_config,
            max_evals=0,  # No new evaluations needed
            metrics_keys=metrics_keys,
            boundaries=instance.boundaries,
            verbose=verbose,
        )

        # Set the loaded data directly on the callback
        instance.callback.solutions = solutions_df
        instance.callback.raw_metrics = scores_df
        instance.callback.pareto_front.front = pareto_df.values.T if len(pareto_df) > 0 else []
        instance.istrain = True

        return instance
