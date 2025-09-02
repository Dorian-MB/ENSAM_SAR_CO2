import sys
from pathlib import Path
from collections import defaultdict
import numbers
import yaml
import pandas as pd
from IPython.display import display

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

from optimizer.utils import get_all_scenarios, calculate_performance_metrics
from eco2_normandy.logger import Logger
from eco2_normandy.simulation import Simulation
from KPIS import Kpis
from KPIS.utils import compute_dynamic_bounds
from colorama import Fore


# dict a des listes comme valeurs par default => defaultdict(list)["nouvelle_cle"].append(valeur) => pas d'erreur
values_by_path = defaultdict(list)


def collect_values(obj, path=""):
    """Collecte toutes les valeurs pour chaque chemin de clé, en généralisant ships et storages."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Si on est sur ships ou storages, on généralise le chemin
            if k in ("ships", "storages") and isinstance(v, list):
                values_by_path[f"{k}.num_{k}"].append(len(v))
                for ship_storages in v:
                    collect_values(ship_storages, path=k)
            else:
                collect_values(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        # Pour les listes qui ne sont pas ships ou storages
        for i, v in enumerate(obj):
            collect_values(v, f"{path}")
    else:
        values_by_path[path].append(obj)


def init_configs(scenarios_path="scenarios/"):
    configs = get_all_scenarios(scenarios_path, ignore_cte=True)
    for _, config in configs:
        collect_values(config)


def get_boundaries():
    # Affiche les bornes pour chaque clé/valeur rencontrée :
    boundaries = {}
    for path, vals in sorted(values_by_path.items()):
        # Filtre les valeurs numériques
        nums = [v for v in vals if isinstance(v, numbers.Number)]
        if nums:
            min_val = float(min(nums))
            max_val = float(max(nums))
            if min_val == max_val:
                boundaries[path] = {"constant": min_val}
            else:
                boundaries[path] = {"min": min_val, "max": max_val}
        else:
            uniques = set(str(v) for v in vals)
            if len(uniques) == 1:
                boundaries[path] = {"constant": list(uniques)[0]}
            else:
                boundaries[path] = {"values": sorted(list(uniques))}
    return boundaries


def get_all_values():
    # Version "liste des valeurs uniques numériques (pas de constantes, pas de strings)"
    print("\nListe des valeurs numériques uniques (hors constantes et strings) :\n")
    unique_values = {}
    for path, vals in sorted(values_by_path.items()):
        # Ne garder que les valeurs numériques
        nums = sorted(set(v for v in vals if isinstance(v, numbers.Number)))
        # On ne garde que si plus d'une valeur numérique
        if len(nums) > 1:
            unique_values[path] = nums
            print(f"{path}: {nums}")
    return unique_values


class ConfigBoundaries:
    """
    Class to store the boundaries of known configuration in `scenarios_path`.
    It loads the boundaries from a YAML file or generates them from the scenarios in the given path.
    It store the boundaries to our model variables
    """

    def __init__(
        self,
        boundaries_yaml="boundaries.yaml",
        scenarios_path="scenarios/",
        logger=None,
        verbose=1,
    ):
        self.log = logger or Logger()
        path = Path.cwd() / "saved" / boundaries_yaml
        if verbose > 0:
            self.log.info(Fore.YELLOW + f"Loading boundaries from {Fore.CYAN + str(path.resolve())}" + Fore.RESET)
        if path.is_file():
            with open(path, "r") as f:
                self._boundaries = yaml.safe_load(f)
                if verbose > 0:
                    self.log.info(Fore.GREEN + f"Boundaries loaded" + Fore.RESET)
        else:
            self.log.info(
                Fore.LIGHTRED_EX
                + f"{boundaries_yaml} not found,{Fore.YELLOW} generating boundaries from scenarios in {scenarios_path}"
                + Fore.RESET
            )
            init_configs(scenarios_path)
            self._boundaries = get_boundaries()
            with open("saved/boundaries.yaml", "w", encoding="utf-8") as f:
                yaml.dump(self._boundaries, f, allow_unicode=True, sort_keys=True)

        self.factory_caps_per_tanks = 5700

        self.ship_capacity = {
            "min": int(self._boundaries["ships.capacity_max"]["min"]),
            "max": int(self._boundaries["ships.capacity_max"]["max"]),
        }
        self.max_num_ships = int(self._boundaries["ships.num_ships"]["max"])
        self.ship_speed = {
            "min": int(self._boundaries["ships.speed_max"]["min"]),
            "max": int(self._boundaries["ships.speed_max"]["max"]),
        }
        self.max_num_storages = int(self._boundaries["storages.num_storages"]["constant"] + 1)
        self.storage_caps = {
            "max": int(self._boundaries["storages.capacity_max"]["max"]),
            "min": int(self._boundaries["storages.capacity_max"]["min"]),
        }
        self.factory_tanks = {
            "max": int(self._boundaries["factory.number_of_tanks"]["max"]),
            "min": int(self._boundaries["factory.number_of_tanks"]["min"]),
        }

        self.initial_destination = 2  # 0=factory, 1=Rotterdam, 2=Bergen
        self.fixed_storage_destination = 1  # 1=Bergen, 0=Rotterdam
        self.verbose = verbose

        # Hidden Variables (Cost)
        self.ship_cost = {
            "max": int(self._boundaries["ships.ship_buying_cost"]["max"]),
            "min": int(self._boundaries["ships.ship_buying_cost"]["min"]),
        }
        self.factory_cost_per_tank = {
            "max": int(self._boundaries["factory.cost_per_tank"]["max"]),
            "min": int(self._boundaries["factory.cost_per_tank"]["min"]),
        }

    def __repr__(self):
        return f"""ConfigBoundaries(max_num_storages={self.max_num_storages}, 
        ship_capacity_min={self.ship_capacity["min"]}, 
        ship_capacity_max={self.ship_capacity["max"]}, 
        max_num_ships={self.max_num_ships}, 
        ship_speed_min={self.ship_speed["min"]}, 
        ship_speed_max={self.ship_speed["max"]})
        factory_tanks_min={self.factory_tanks["min"]}, 
        factory_tanks_max={self.factory_tanks["max"]},
        storage_caps_min={self.storage_caps["min"]},
        storage_caps_max={self.storage_caps["max"]}
        """


class KpisBoundaries:
    def __init__(
        self,
        kpis_boundaries_file: str = "kpis_bounds.csv",
        verbose: int = 1,
        logger=None,
    ):
        self.verbose = verbose
        self.log = logger or Logger()
        self.path = Path.cwd() / "saved" / kpis_boundaries_file
        if self.path.exists() and self.path.is_file():
            if verbose > 0:
                self.log.info(
                    Fore.GREEN + f"Loading KPIs boundaries from {Fore.CYAN + str(self.path.resolve())}" + Fore.RESET
                )
            self.kpis_boundaries = pd.read_csv(str(self.path), index_col="bounds").T
        else:
            if verbose > 0:
                self.log.info(
                    Fore.LIGHTRED_EX
                    + f"{kpis_boundaries_file} not found, generating KPIs boundaries from scenarios"
                    + Fore.RESET
                )
            self.kpis_boundaries = self._compute_and_save_kpis_boundaries()


    def _compute_and_save_kpis_boundaries(self):
        """
        Compute the boundaries for KPIs from the scenarios.
        """
        scenarios = get_all_scenarios("scenarios/")
        self._kpis_list = []
        for i, (path, config) in enumerate(scenarios):
            config["general"]["num_period"] = 2000  # Set to 2000 for reproducibility and stable kpis results
            sim = Simulation(config, verbose=False)
            sim.run()
            self._kpis_list.append(calculate_performance_metrics(config, sim).to_dict(orient="records")[0])
            if self.verbose > 1:
                self.log.info(Fore.CYAN + str(i) + f" {path.name}" + Fore.RESET)
                display(pd.DataFrame(self._kpis_list[-1], index=[0]))
                print()

        print(self._kpis_list)
        bounds = compute_dynamic_bounds(self._kpis_list)
        bounds_df = pd.DataFrame(bounds, index=["min", "max"])
        bounds_df.index.name = "bounds"
        bounds_df.underfill_rate = bounds_df.underfill_rate.clip(lower=0, upper=1)  # Clip underfilling rate to [0, 1]
        bounds_df.to_csv(self.path, index=True)
        return bounds_df

    def __repr__(self):
        return f"""KpisBoundaries(
            kpis_bounds={self.kpis_boundaries.to_dict(orient="index")}
            )"""


def get_kpis_boundaries() -> pd.DataFrame:
    return KpisBoundaries(verbose=0).kpis_boundaries


if __name__ == "__main__":
    from pprint import pprint
    from IPython.display import display

    init_configs()

    unique_values = get_all_values()
    with open("saved/boundaries_range.yaml", "w", encoding="utf-8") as f:
        yaml.dump(unique_values, f, allow_unicode=True, sort_keys=True)
    print("\nLes valeurs numériques uniques ont été sauvegardées dans boundaries_range.yaml")

    cfg = ConfigBoundaries()
    print(cfg)

    bounds = KpisBoundaries(verbose=2)
    display(bounds.kpis_boundaries)
