from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from eco2_normandy.logger import Logger
from eco2_normandy.port import Port


class Storage(Port):
    def __init__(
        self,
        num_period_per_hours,
        env,
        name,
        capacity_max,
        consumption_rate,
        maintenance_rate,
        docks,
        pbs_to_dock,
        pump_rate,
        pump_in_maintenance_rate,
        lock_waiting_time,
        unloading_time,
        transit_time_to_dock,
        transit_time_from_dock,
        storage_cost_per_m3,
        number_of_tanks,
        cost_per_tank,
        logger,
        **kwargs
    ) -> None:
        super().__init__(
            num_period_per_hours,
            env,
            name,
            capacity_max,
            docks,
            pbs_to_dock,
            pump_rate=pump_rate,
            pump_in_maintenance_rate=pump_in_maintenance_rate,
            number_of_tanks=number_of_tanks,
            cost_per_tank=cost_per_tank,
            logger=logger,
        )
        self.logger = logger or Logger()
        self.consumption_rate: int = consumption_rate
        self.unloading_time = unloading_time  # temps fixe de déchargement
        self.transit_time_to_dock = transit_time_to_dock
        self.transit_time_from_dock = transit_time_from_dock
        self.storage_cost_per_m3 = storage_cost_per_m3
        self.received_co2_over_time = 0
        self.__dict__.update(kwargs)
        self.action = env.process(self.run())

    def _save_state(self):
        states_to_save = [
            "capacity",
            "capacity_max",
            "consumption_rate",
            "maintenance",
            "end_of_maintenance",
            "dock_usage",
            "received_co2_over_time",
            "storage_cost_per_m3",
        ]
        self.history.append({k: getattr(self, k) for k in states_to_save})

    def run(self):
        while True:
            consumed = min(self.consumption_rate, self.capacity)
            if consumed:
                yield self.container.get(consumed)
            yield self.env.timeout(1)

    def pump(self, amount):
        pump_rate = (
            self.pump_in_maintenance_rate if self.maintenance else self.pump_rate
        )
        available_space = self._get_capacity_left()
        transferable = min(pump_rate, amount, available_space)
        if transferable:
            yield self.container.put(transferable)
        self.received_co2_over_time += transferable
        return transferable

    def unload_co2(self, quantity: int):
        available_space = self._get_capacity_left()
        to_add = min(quantity, available_space)
        if to_add:
            yield self.container.put(to_add)
        excess = quantity - to_add
        return excess
