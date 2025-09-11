import pandas as pd


def flatten(lst: list) -> list:
    return [item for elem in lst for item in (flatten(elem) if isinstance(elem, list) else [elem])]


## Pareto ####################################################################
def dominates(m1: dict, m2: dict) -> bool:
    """Renvoie True si m1 domine m2 (tous les indicateurs ≤ et au moins un <)."""
    return all(m1[k] <= m2[k] for k in m1) and any(m1[k] < m2[k] for k in m1)


class ParetoFront:
    def __init__(self, metrics_keys: list[str] = None):
        self.front: list[tuple] = []
        self.metrics_keys = metrics_keys

    def add(self, metrics: dict | pd.Series, n_evals: int = None):
        """Ajoute (sol,metrics) au front si non dominé, et retire les solutions dominées."""
        non_dom = []
        if isinstance(metrics, pd.Series):
            metrics = metrics.to_dict()
        if self.metrics_keys:
            assert all(k in self.metrics_keys for k in metrics), "All metrics must be in the defined keys."
        for entry, n in self.front:
            if dominates(entry, metrics):  # Si entry domine la nouvelle → on ne l'ajoute pas
                return  # on garde les meme solutions -> on quitte
            if not dominates(metrics, entry):
                non_dom.append((entry, n))  # ni l'un ni l'autre ne domine → on garde entry
            # si entry est dominer par la nouvelle solution, on ne la garde pas
        # on n'a pas quitté → la nouvelle solution est non dominée
        non_dom.append((metrics, n_evals))
        self.front = non_dom

    def create(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        for _, row in data.iterrows():
            self.add(row)
        return pd.DataFrame([entry for entry, _ in self.front])

    def is_dominated(self, metrics: dict) -> bool:
        """Vérifie si la solution metrics est dominée par une solution du front."""
        for entry, _ in self.front:
            if dominates(entry, metrics):
                return True
        return False

    def get_front(self):
        return {f"solution_{n or i}": sol for i, (sol, n) in enumerate(self.front)}


def get_paretofront(data: pd.DataFrame) -> pd.DataFrame:
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
    inv = (
        cfg["factory"]["cost_per_tank"] * cfg["factory"]["number_of_tanks"]
        + cfg["ships"][0]["ship_buying_cost"] * n_ships
    )
    # Operating cost approx: fuel_price × speed × #ships
    fuel_price = cfg["KPIS"]["fuel_price_per_ton"]
    op = fuel_price * speed * n_ships
    cost = inv + op

    # Wasted CO2 approx (volume minus transport capacity)
    hours_in_year = 24 * 365
    prod_rate = cfg["factory"]["sources"][0]["annual_production_capacity"] / (
        hours_in_year * cfg["general"]["num_period_per_hours"]
    )
    trans_cap = sol["ship_capacity"] * n_ships
    wasted = max(0, prod_rate * total_period - trans_cap)

    # Pour chaque navire
    waiting = 0
    cumulative_travel_time = 0
    for i in range(n_ships):
        dest = sol[f"fixed{i + 1}_storage_destination"]
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
        dest = sol[f"fixed{i + 1}_storage_destination"]
        distance = cfg["general"]["distances"]["Le Havre"][map_ship_fixed_storage_destination[dest]]
        travel_time = distance / speed
        voyages = total_period // travel_time
        transported = voyages * sol["ship_capacity"]
        # Si la production dépasse ce que ce navire peut transporter, le surplus attend
        produced = prod_rate * total_period / n_ships
        waiting += max(0, produced - transported)

    return {"cost": cost, "wasted": wasted, "waiting": waiting, "travel": travel}
