from abc import ABC, abstractmethod
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from eco2_normandy.logger import Logger
from simpy import Resource, Container


# --- Classe Port avec simpy.Resource pour la gestion des quais et simpy.Container pour la capaciter de co2 ---
class Port(ABC):
    def __init__(
        self,
        num_period_per_hours: float,
        env,
        name,
        capacity_max,
        docks,
        pbs_to_dock,
        number_of_tanks: int,
        cost_per_tank: int,
        pump_rate,
        pump_in_maintenance_rate,
        lock_waiting_time: int = 0,
        initial_capacity: int = 0,
        logger=None,
        **kwargs
    ):
        self.env = env
        self.name = name
        self.logger = logger or Logger()
        # self.logger.info(f"Creating port {self.name} with capacity {capacity_max} and docks {docks}")

        # Paramètres
        self.num_period_per_hours = num_period_per_hours
        self.capacity_max: int = capacity_max
        self.docks: int = docks
        self.pbs_to_dock = pbs_to_dock / num_period_per_hours
        self.number_of_tanks = number_of_tanks
        self.cost_per_tank = cost_per_tank
        self.pump_rate: int = pump_rate * num_period_per_hours
        self.pump_in_maintenance_rate: int = (
            pump_in_maintenance_rate * num_period_per_hours
        )
        self.lock_waiting_time: int = lock_waiting_time

        # Variables d'état
        self.history: list = []
        self.maintenance: bool = False
        self.end_of_maintenance: int = 0
        self.wasted_production: int = 0
        self._counter = 0

        # Variables d'état
        init_val = min(initial_capacity, capacity_max)
        self.container = Container(env, init=init_val, capacity=capacity_max)

        # Gestion du quai avec simpy.Resource
        self.dock_resource = Resource(env, capacity=self.docks)

    @property
    def dock_usage(self):
        return self.dock_resource.count

    @property
    def capacity(self):
        return self.container.level

    def request(self):
        req = self.dock_resource.request()
        req.order = self._counter
        self._counter += 1
        return req

    def release(self, req):
        self._counter -= 1
        return self.dock_resource.release(req)

    def _check_tanks_capacity(self, quantity) -> bool:
        return (self.capacity + quantity) <= self.capacity_max

    def _get_capacity_left(self):
        return self.capacity_max - self.capacity

    def load_to_tank(self, quantity):
        """Ajoute une quantité à la capacité et retourne l’excès (gaspillé)
        Sans blocage : on ajoute ce qui rentre dans l’espace libre."""
        available = self._get_capacity_left()
        to_add = min(quantity, available)
        if to_add:
            yield self.container.put(to_add)
        else:
            excess = quantity - to_add
            yield excess

    def _get_pbs_to_dock(self):
        return self.pbs_to_dock

    def _lock_available(self):
        return True

    def __str__(self):
        return self.name
