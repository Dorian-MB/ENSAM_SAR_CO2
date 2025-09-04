from copy import deepcopy
from pathlib import Path
from typing import Generator
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import pandas as pd

from eco2_normandy.simulation import Simulation
from eco2_normandy.tools import get_simlulation_variable
from KPIS import Kpis


metrics_keys = ["cost", "wasted_production_over_time", "waiting_time", "underfill_rate"]


def calculate_performance_metrics(cfg, sim, metrics_keys=metrics_keys):
    """Évalue la configuration en lançant la simulation."""
    dfs = sim.result
    kpis = Kpis(dfs, cfg)
    functional_cost = kpis.calculate_functional_kpis()
    cost = functional_cost["Combined Total Cost"]
    wasted_production_over_time = kpis.wasted_production()
    waiting_time = kpis.get_total_waiting_time()
    factory_filling_rate = kpis.factory_filling_rate()  # want to maximize, so we will use -factory_filling_rate
    metrics = {
        k: v
        for k, v in zip(
            metrics_keys,
            [cost, wasted_production_over_time, waiting_time, 1 - factory_filling_rate],
        )
    }
    return pd.DataFrame(metrics, index=[0])


def get_all_scenarios(path: str, ignore_cte=False) -> Generator[tuple[Path, dict], None, None]:
    """
    Récupère tous les scénarios à partir d'un fichier YAML.
    """
    ignore_keys = {"KPIS", "general", "allowed_speeds", "weather_probability"}
    for path in Path(path).glob("**/*.yaml"):
        config = get_simlulation_variable(str(path))[0]
        if ignore_cte:
            for key in ignore_keys:
                config.pop(key, None)
        if path.is_file() and path.suffix == ".yaml":
            config["name"] = path.stem
            if not ignore_cte:
                config["general"]["num_period"] = 2_000
            yield path, config


def evaluate_single_scenario(scenario: dict, return_score: bool = True) -> pd.DataFrame:
    """Evaluate a scenario.

    Args:
        scenario (dict): The scenario configuration.

    Returns:
        pd.DataFrame: The evaluation results.
    """
    # Create a simulation instance
    sim = Simulation(scenario, verbose=0)
    sim.run()
    result = calculate_performance_metrics(scenario, sim)
    if return_score:
        if not hasattr(evaluate_single_scenario, "normalize"):
            evaluate_single_scenario.normalize = Normalizer()
        norm_df = evaluate_single_scenario.normalize(result)
        result["score"] = evaluate_single_scenario.normalize.compute_score(norm_df)
    return result


metrics_keys: list[str] = [
    "cost",
    "wasted_production_over_time",
    "waiting_time",
    "underfill_rate",
]
metrics_weight: list[int] = [20, 20, 10, 15]


class Normalizer:
    """Normalize KPI values."""

    def __init__(
        self,
        kpis_boundaries: pd.DataFrame = None,
        metrics_keys=metrics_keys,
        metrics_weight=metrics_weight,
    ):
        self.metrics_keys = metrics_keys
        self.metrics_weight = metrics_weight
        if kpis_boundaries:
            self.kpis_boundaries = kpis_boundaries
        else:
            from optimizer.boundaries import get_kpis_boundaries

            self.kpis_boundaries = get_kpis_boundaries()

    def normalize(self, kpis_list: pd.DataFrame, clip: bool = False) -> pd.DataFrame:
        """Normalize the KPIs using the provided boundaries."""
        normalized = (kpis_list - self.kpis_boundaries["min"]) / (
            self.kpis_boundaries["max"] - self.kpis_boundaries["min"]
        )
        if clip:
            return normalized.clip(0, 1)
        return normalized

    def compute_score(self, norm_df: pd.DataFrame):
        """Compute the score for the KPIs."""
        weighted = norm_df * self.metrics_weight
        return weighted.sum(axis=1)

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)


########## Usefull classes ##########
# Config builder for simulation
class ConfigBuilderFromSolution:
    def __init__(self, base_config: dict, boundaries=None):
        self.base_config = base_config
        self.map_ship_initial_destination = {0: "Le Havre", 1: "Rotterdam", 2: "Bergen"}
        self.map_ship_fixed_storage_destination = {0: "Rotterdam", 1: "Bergen"}
        if boundaries is None:
            from optimizer.boundaries import Boundaries

            self.boundaries = Boundaries()
        else:
            self.boundaries = boundaries

    def get_config_from_solution(self, sol: dict, algorithm: str, *args, **kwargs) -> dict:
        """
        Build a simulation config from a solution dict.
        """
        if algorithm in ("heuristic", "HeuristicSolver"):
            return self.build_heuristic(sol)
        else:
            return self.build(sol)

    def predict_cost(self, x: int, X: tuple, Y: tuple) -> int:
        """Predict the cost for a given input.

        Args:
            x (int): value to predict
            X (tuple): bounds: min[0]/max[1] of tank or ship caps
            Y (tuple): bound: min[0]/max[1] of cost for tank or ship

        Returns:
            int: cost prediction
        """
        pente = (Y[1] - Y[0]) / (X[1] - X[0])
        ordonnee = Y[0] - pente * X[0]
        return pente * x + ordonnee

    def _get_storage_name(self, sol: dict, i: int) -> str:
        if sol["use_Bergen"] and sol["use_Rotterdam"]:
            return ["Bergen", "Rotterdam"][i]
        elif sol["use_Bergen"]:
            return "Bergen"
        elif sol["use_Rotterdam"]:
            return "Rotterdam"

    def build(self, sol: dict, num_period: int = 2000) -> dict:
        cfg = deepcopy(self.base_config)
        cfg["factory"]["number_of_tanks"] = sol["number_of_tanks"]
        cfg["factory"]["capacity_max"] = int(sol["number_of_tanks"] * self.boundaries.factory_caps_per_tanks)
        X = (self.boundaries.factory_tanks["min"], self.boundaries.factory_tanks["max"])
        Y = (
            self.boundaries.factory_cost_per_tank["min"],
            self.boundaries.factory_cost_per_tank["max"],
        )
        cfg["factory"]["cost_per_tank"] = self.predict_cost(sol["number_of_tanks"], X, Y)
        cfg["factory"]["intial_capacity"] = 0

        storage = deepcopy(cfg["storages"][0])
        storage["name"] = ""
        storage["capacity_max"] = sol["storage_caps"]
        cfg["storages"].clear()
        for i in range(sol["num_storages"]):
            cfg["storages"].append(deepcopy(storage))
            cfg["storages"][i]["name"] = self._get_storage_name(sol, i)

        # add initial port
        X = (self.boundaries.ship_capacity["min"], self.boundaries.ship_capacity["max"])
        Y = (self.boundaries.ship_cost["min"], self.boundaries.ship_cost["max"])
        ship = deepcopy(cfg["ships"][0])
        cfg["ships"].clear()
        for i in range(sol["num_ship"]):
            cfg["ships"].append(deepcopy(ship))
            cfg["ships"][i]["name"] = f"Ship {i + 1}"
            cfg["ships"][i]["init"]["destination"] = self.map_ship_initial_destination[sol[f"init{i + 1}_destination"]]
            cfg["ships"][i]["fixed_storage_destination"] = self.map_ship_fixed_storage_destination[
                sol[f"fixed{i + 1}_storage_destination"]
            ]
            cfg["ships"][i]["capacity_max"] = sol["ship_capacity"]
            cfg["ships"][i]["speed_max"] = sol["ship_speed"]
            cfg["ships"][i]["ship_buying_cost"] = self.predict_cost(sol["ship_capacity"], X, Y)
        cfg["general"]["number_of_ships"] = sol["num_ship"]
        cfg["general"]["num_period"] = num_period
        return cfg

    def build_heuristic(self, sol: dict) -> dict:
        """Build surrogate config from solution."""
        cfg = deepcopy(self.base_config)
        for i in range(sol["num_ship"]):
            cfg["ships"][i]["capacity_max"] = sol[f"ship_capacity"]
            cfg["ships"][i]["speed_max"] = sol[f"ship_speed"]
        cfg["general"]["number_of_ships"] = sol["num_ship"]
        return cfg


# No profiler class (cprofile)
class NoProfiler:
    """
    A no-op profiler class that does nothing.
    This is used when profiling is disabled.
    """

    def enable(self):
        pass

    def disable(self):
        pass

    def print_stats(self):
        pass
