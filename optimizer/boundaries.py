import sys
from pathlib import Path
from collections import defaultdict
import numbers
import yaml
sys.path.append(str(Path.cwd()))
from optimizer.utils import get_all_scenarios
from eco2_normandy.logger import Logger
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
    def __init__(self, boundaries_yaml="boundaries.yaml", scenarios_path="scenarios/", logger=None):
        self.log = logger or Logger()
        path = Path.cwd() / "optimizer" / boundaries_yaml
        self.log.info(Fore.YELLOW+f"Loading boundaries from {Fore.CYAN+str(path.resolve())}"+Fore.RESET)
        if path.is_file():
            with open(path, 'r') as f:
                self.boundaries = yaml.safe_load(f)
                self.log.info(Fore.GREEN+f"Boundaries loaded"+Fore.RESET)
        else:
            self.log.info(Fore.LIGHTRED_EX+f"{boundaries_yaml} not found,{Fore.YELLOW} generating boundaries from scenarios in {scenarios_path}"+Fore.RESET)
            init_configs(scenarios_path)
            self.boundaries = get_boundaries()

        self.max_num_storages = int(self.boundaries.get("storages.num_storages")["constant"] + 1)
        self.ship_capacity_min = int(self.boundaries.get("ships.capacity_max")["min"])
        self.ship_capacity_max = int(self.boundaries.get("ships.capacity_max")["max"])
        self.max_num_ships = int(self.boundaries.get("ships.num_ships")["max"])                 
        self.ship_speed_min, self.ship_speed_max = int(self.boundaries.get("ships.speed_max")["min"]), int(self.boundaries.get("ships.speed_max")["max"])
        self.initial_destination = 2 # 0=factory, 1=Rotterdam, 2=Bergen 
        self.fixed_storage_destination = 1  # 1=Bergen, 0=Rotterdam 
        
    def __repr__(self):
        return f"""ConfigBoundaries(max_num_storages={self.max_num_storages}, 
        ship_capacity_min={self.ship_capacity_min}, 
        ship_capacity_max={self.ship_capacity_max}, 
        max_num_ships={self.max_num_ships}, 
        ship_speed_min={self.ship_speed_min}, 
        ship_speed_max={self.ship_speed_max})"""


if __name__ == "__main__":

    init_configs()

    boundaries = get_boundaries()
    with open("solver/boundaries.yaml", "w", encoding="utf-8") as f:
        yaml.dump(boundaries, f, allow_unicode=True, sort_keys=True)
    print("\nLes bornes ont été sauvegardées dans boundaries.yaml")

    unique_values = get_all_values()
    with open("solver/boundaries_range.yaml", "w", encoding="utf-8") as f:
        yaml.dump(unique_values, f, allow_unicode=True, sort_keys=True)
    print("\nLes valeurs numériques uniques ont été sauvegardées dans boundaries_range.yaml")

    cfg = ConfigBoundaries()
    print(cfg)