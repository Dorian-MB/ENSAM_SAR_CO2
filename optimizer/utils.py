from copy import deepcopy
from pathlib import Path
from typing import Generator
import sys
sys.path.append(str(Path.cwd()))

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))    

from eco2_normandy.tools import get_simlulation_variable
from KPIS import Kpis

def flatten(lst): 
    return [item for elem in lst for item in (flatten(elem) if isinstance(elem, list) else [elem])]

## Pareto ####################################################################
def dominates(m1: dict, m2: dict) -> bool:
    """Renvoie True si m1 domine strictement m2 sur tous les KPI."""
    # tous les indicateurs ≤ et au moins un < 
    return all(m1[k] <= m2[k] for k in m1) and any(m1[k] < m2[k] for k in m1)

class ParetoFront:
    def __init__(self, metrics_keys:list[str]=None):
        self.front:list[tuple] = [] 
        self.metrics_keys = metrics_keys

    def add(self, metrics:dict|pd.Series, n_evals:int=None):
        """Ajoute (sol,metrics) au front si non dominé, et retire les solutions dominées."""
        non_dom = []
        if isinstance(metrics, pd.Series): 
            metrics = metrics.to_dict()
        if self.metrics_keys:
            assert all(k in self.metrics_keys for k in metrics), "All metrics must be in the defined keys."
        for entry, n in self.front:
            if dominates(entry, metrics): # Si entry domine la nouvelle → on ne l'ajoute pas
                return # on garde les meme solutions -> on quitte
            if not dominates(metrics, entry):
                non_dom.append((entry, n))# ni l'un ni l'autre ne domine → on garde entry
            # si entry est dominer par la nouvelle solution, on ne la garde pas
        # on n'a pas quitté → la nouvelle solution est non dominée
        non_dom.append((metrics, n_evals))
        self.front = non_dom

    def create(self, data:pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        for _, row in data.iterrows():
            self.add(row)
        return pd.DataFrame([entry for entry, _ in self.front])

    def is_dominated(self, metrics):
        """Vérifie si la solution metrics est dominée par une solution du front."""
        for entry, _ in self.front:
            if dominates(entry, metrics):
                return True
        return False

    def get_front(self):
        return {f"solution_{n or i}":sol for i, (sol, n) in enumerate(self.front)}

def get_paretofront(data:pd.DataFrame)->pd.DataFrame:
    """
    Retourne le front de Pareto en considérant tous les KPI comme à
    *minimiser*. Si vous avez des KPIs à maximiser, il suffit de
    les transformer (ex: sales → -sales) avant d'appeler cette fonction.
    """
    return ParetoFront().create(data)

## Surrogate metrics ####################################################################
def surrogate_metrics(sol, cfg):
    map_ship_fixed_storage_destination = {1: "Bergen", 0: "Rotterdam"}
    total_period = cfg["general"]["num_period"] 
    speed = sol["ship_speed"]
    n_ships = sol["num_ship"]
    # Investment approx
    inv = (cfg["factory"]["cost_per_tank"] * cfg["factory"]["number_of_tanks"]
          + cfg["ships"][0]["ship_buying_cost"] * n_ships)
    # Operating cost approx: fuel_price × speed × #ships
    fuel_price = cfg["KPIS"]["fuel_price_per_ton"]
    op = fuel_price * speed * n_ships
    cost = inv + op

    # Wasted CO2 approx (volume minus transport capacity)
    hours_in_year = 24 * 365
    prod_rate =  cfg["factory"]["sources"][0]["annual_production_capacity"] / (hours_in_year * cfg["general"]["num_period_per_hours"])
    trans_cap = sol["ship_capacity"] * n_ships
    wasted = max(0, prod_rate * total_period - trans_cap)

    # Pour chaque navire
    waiting = 0
    cumulative_travel_time = 0
    for i in range(n_ships):
        dest = sol[f"fixed{i+1}_storage_destination"]
        distance = cfg["general"]["distances"]["Le Havre"][map_ship_fixed_storage_destination[dest]]
        travel_time = distance / speed
        cumulative_travel_time += travel_time
        voyages = total_period // travel_time
        transported = voyages * sol["ship_capacity"]
        produced = prod_rate * total_period / n_ships
        waiting += max(0, produced - transported)
    travel = total_period // cumulative_travel_time 

    # Approximate waiting proxy
    for i in range(n_ships):
        dest = sol[f"fixed{i+1}_storage_destination"]
        distance = cfg["general"]["distances"]["Le Havre"][map_ship_fixed_storage_destination[dest]]
        travel_time = distance / speed
        voyages = total_period // travel_time
        transported = voyages * sol["ship_capacity"]
        # Si la production dépasse ce que ce navire peut transporter, le surplus attend
        produced = prod_rate * total_period / n_ships
        waiting += max(0, produced - transported)

    return {"cost": cost,
            "wasted": wasted,
            "waiting": waiting,
            "travel": travel}

metrics_keys = ["cost", "wasted_production_over_time", "waiting_time", "underfill_rate"]
def calculate_performance_metrics(cfg, sim, metrics_keys=metrics_keys):
        """Évalue la configuration en lançant la simulation."""
        dfs = sim.result 
        kpis = Kpis(dfs, cfg)
        functional_cost = kpis.calculate_functional_kpis()
        cost = functional_cost["Combined Total Cost"]
        wasted_production_over_time = kpis.wasted_production()
        waiting_time = kpis.get_total_waiting_time()
        factory_filling_rate = kpis.factory_filling_rate() # want to maximize, so we will use -factory_filling_rate
        metrics = { k: v for k, v in zip(metrics_keys,
                                         [cost, wasted_production_over_time, waiting_time, 1-factory_filling_rate])}
        return metrics

def get_all_scenarios(path: str, ignore_cte=False) -> Generator[dict, None, None]:
    """
    Récupère tous les scénarios à partir d'un fichier YAML.
    """
    ignore_keys = {"KPIS", "general", "allowed_speeds", "weather_probability"}
    for path in Path(path).glob("**/*.yaml"):
        config = get_simlulation_variable(str(path))[0]
        if ignore_cte:
            for key in ignore_keys:
                config.pop(key, None)
        if path.is_file() and path.suffix == '.yaml':
            yield path, config


########## Usefull classes ##########
# Config builder for simulation 
class ConfigBuilderFromSolution:
    def __init__(self, base_config, sol=None):
        self.base_config = base_config
        self.map_ship_initial_destination = {0: "Le Havre", 1: "Rotterdam", 2: "Bergen"}
        self.map_ship_fixed_storage_destination = {0: "Rotterdam", 1: "Bergen"}
        if sol is not None:
                self.cfg = self.build(sol)
        else:
                self.cfg = None


    def get_config_from_solution(self, sol:dict, algorithm:str, *args, **kwargs) -> dict:
        """
        Build a simulation config from a solution dict.
        """
        if algorithm in ("heuristic", "HeuristicSolver"):
            return self.build_heuristic(sol)
        else :
            return self.build(sol)
        

    def _get_storage_name(self, sol, i):
        if sol["use_Bergen"] and sol["use_Rotterdam"]:
            return ["Bergen", "Rotterdam"][i]
        elif sol["use_Bergen"]:
            return "Bergen"
        elif sol["use_Rotterdam"]:
            return "Rotterdam"
        
    def build(self, sol:dict)->dict:
        cfg = deepcopy(self.base_config)
        base_factory = cfg["factory"]
        cfg["factory"]["number_of_tanks"] = sol.get("number_of_tanks", base_factory["number_of_tanks"])
        cfg["factory"]["capacity_max"] = sol.get("capacity_max_factory", base_factory["capacity_max"])
        cfg["factory"]["docks"] = sol.get("factory_docks", base_factory["docks"]) 

        storage = deepcopy(cfg["storages"][0])
        storage["name"] = "" 
        cfg["storages"].clear()
        for i in range(sol.get("num_storages", len(cfg["storages"]))):
            cfg["storages"].append(deepcopy(storage))
            cfg["storages"][i]["name"] = self._get_storage_name(sol, i)

        # add initial port
        ship = deepcopy(cfg["ships"][0])
        cfg["ships"].clear()
        for i in range(sol["num_ship"]):
            cfg["ships"].append(deepcopy(ship))
            cfg["ships"][i]["name"] = f"Ship {i+1}"
            cfg["ships"][i]["init"]["destination"] = self.map_ship_initial_destination[sol[f"init{i+1}_destination"]]
            cfg["ships"][i]["fixed_storage_destination"] = self.map_ship_fixed_storage_destination[sol[f"fixed{i+1}_storage_destination"]]
            cfg["ships"][i]["capacity_max"] = sol[f"ship_capacity"]
            cfg["ships"][i]["speed_max"] = sol[f"ship_speed"]  
        cfg["general"]["number_of_ships"] = sol["num_ship"]
        return cfg

    def build_heuristic(self, sol:dict) -> dict:
        """Build surrogate config from solution."""
        cfg = deepcopy(self.base_config)
        for i in range(sol["num_ship"]):
            cfg["ships"][i]["capacity_max"] = sol[f"ship_capacity"]
            cfg["ships"][i]["speed_max"] = sol[f"ship_speed"]
        cfg["general"]["number_of_ships"] = sol["num_ship"]
        return cfg

    def decode_and_repair(self, x, max_ships):
        """
        Decode the decision vector x into a solution dict and enforce storage consistency.
        """
        sol = {}
        idx = 0
        # High-level variables
        sol['num_storages']       = int(x[idx]); idx += 1
        sol['use_Bergen']         = int(x[idx]); idx += 1
        sol['use_Rotterdam']      = int(x[idx]); idx += 1
        sol['num_ship']           = int(x[idx]); idx += 1
        sol['ship_capacity']      = int(x[idx]); idx += 1
        sol['ship_speed']         = int(x[idx]); idx += 1
        # Per-ship destinations
        for i in range(max_ships):
            sol[f'init{i+1}_destination'] = int(x[idx]); idx += 1
        for i in range(max_ships):
            sol[f'fixed{i+1}_storage_destination'] = int(x[idx]); idx += 1

        # Repair storage flags based on num_storages and per-ship destinations
        if sol['num_storages'] >= 2:
            sol['use_Bergen']    = 1
            sol['use_Rotterdam'] = 1
        else:
            flags = [sol[f'fixed{i+1}_storage_destination'] for i in range(max_ships)]
            if any(f == 1 for f in flags):
                sol['use_Bergen']    = 1
                sol['use_Rotterdam'] = 0
            else:
                sol['use_Bergen']    = 0
                sol['use_Rotterdam'] = 1
            sol['num_storages'] = sol['use_Bergen'] + sol['use_Rotterdam']

        return sol

# Logger for multiprocessing (base logging python package throw error while multiprocessing)
class LoggerForMultiprocessing:
    """
    Substitue à un logger pour les environnements de multiprocessing.
    Logger classique ne support pas le multiprocessing.
    """

    def debug(self, message: str):
        print("DEBUG:" + message)

    def info(self, message: str):
        print("INFO: " + message)

    def error(self, message: str):
        print("ERROR: " + message)

    def warning(self, message: str):
        print("WARNING: " + message)
    
    def critical(self, message: str):
        print("CRITICAL: " + message)
    
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