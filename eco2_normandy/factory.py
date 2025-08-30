import random

from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from eco2_normandy.logger import Logger
from eco2_normandy.port import Port

NUM_HOURS_IN_YEARS = 24 * 365


class Factory(Port):
    """
    Factory class representing a CO2 production facility with storage and shipping capabilities.

    Inherits from Port class and simulates factory operations including:
    - CO2 production from multiple sources
    - Storage tank management
    - Maintenance events (scheduled and unscheduled)
    - Ship loading operations
    - Port logistics (pilots, weather delays, etc)

    Args:
        num_period_per_hours (int): Number of simulation periods per hour
        env (simpy.Environment): SimPy simulation environment
        name (str): Name identifier for the factory
        capacity_max (float): Maximum storage capacity in tons
        sources (list): List of production sources with quantities and maintenance rates
        docks (int): Number of available docks
        pbs_to_dock (int): time from PBS to dock
        scheduled_maintenance_period_major (int): Period between major maintenance events
        scheduled_maintenance_period_minor (int): Period between minor maintenance events
        unscheduled_maintenance_prob (float): Probability of unscheduled maintenance per period
        pump_rate (float): Normal CO2 transfer rate when pumping to ships
        pump_in_maintenance_rate (float): Reduced pump rate during maintenance
        loading_time (float): Base time required for loading operations
        weather_waiting_time (float): Delay time due to weather
        pilot_waiting_time (float): Wait time for pilot availability
        dock_waiting_time (float): Wait time for dock availability
        lock_waiting_time (float): Wait time for lock operations
        initial_capacity (float): Starting storage capacity
        transit_time_to_dock (float): Time for ship to reach dock
        transit_time_from_dock (float): Time for ship to depart dock
        number_of_tanks (int): Number of storage tanks
        cost_per_tank (float): Cost per storage tank


    """

    def __init__(
        self,
        num_period_per_hours,
        env,
        name,
        capacity_max,
        sources,
        docks,
        pbs_to_dock,
        scheduled_maintenance_period_major,
        scheduled_maintenance_period_minor,
        unscheduled_maintenance_prob,
        pump_rate,
        pump_in_maintenance_rate,
        loading_time,
        weather_waiting_time,
        pilot_waiting_time,
        dock_waiting_time,
        lock_waiting_time,
        initial_capacity,
        transit_time_to_dock,
        transit_time_from_dock,
        number_of_tanks,
        cost_per_tank,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            num_period_per_hours,
            env,
            name,
            capacity_max,
            docks,
            pbs_to_dock=pbs_to_dock,
            pump_rate=pump_rate,
            pump_in_maintenance_rate=pump_in_maintenance_rate,
            number_of_tanks=number_of_tanks,
            cost_per_tank=cost_per_tank,
            initial_capacity=initial_capacity,
            logger=logger,
        )
        self.logger = logger or Logger()
        self.sources = sources
        # Maintenance programmée majeure (impact de -1.5%)
        self.scheduled_maintenance_period_major = scheduled_maintenance_period_major
        self.maintenance_impact_major = -0.015
        # Maintenance programmée mineure (impact de -0.5%)
        self.scheduled_maintenance_period_minor = scheduled_maintenance_period_minor
        self.maintenance_impact_minor = -0.005
        # Maintenance non programmée (impact de -0.01%)
        self.unscheduled_maintenance_prob = unscheduled_maintenance_prob
        self.maintenance_impact_unscheduled = -0.01

        # Temps d'attente et de chargement (conversion selon le nombre de périodes par heure)
        self.loading_time = loading_time / num_period_per_hours
        self.weather_waiting_time = weather_waiting_time / num_period_per_hours
        self.pilot_waiting_time = pilot_waiting_time / num_period_per_hours
        self.dock_waiting_time = dock_waiting_time / num_period_per_hours
        self.lock_waiting_time = lock_waiting_time / num_period_per_hours

        # Temps de transit
        self.transit_time_to_dock = transit_time_to_dock / num_period_per_hours
        self.transit_time_from_dock = transit_time_from_dock / num_period_per_hours

        # var for production
        self._maintenance_counters = [0] * len(self.sources)
        self._maintenance_flags = [False] * len(self.sources)

        self.__dict__.update(kwargs)
        self.action = env.process(self.run())

    def _save_state(self):
        states_to_save = [
            "capacity",
            "wasted_production",
            "maintenance",
            "dock_usage",
            "capacity_max",
            "number_of_tanks",
            "cost_per_tank",
        ]
        self.history.append({k: getattr(self, k) for k in states_to_save})

    def _generate_production(self):
        """
        Helper to generate the production of the factory
        """

        production_rate = 0
        for i, source in enumerate(self.sources):
            if self._maintenance_counters[i] == 0:
                self._maintenance_flags[i] = random.random() < source["maintenance_rate"]

            # Si aucune maintenance n'est active durant l'heure, on ajoute la production
            if not self._maintenance_flags[i]:
                # Production pour la période = production horaire / nombre de périodes par heure.
                production_rate += source["annual_production_capacity"] / (
                    NUM_HOURS_IN_YEARS * self.num_period_per_hours
                )

            # Passage à la période suivante dans l'heure pour cette source
            self._maintenance_counters[i] = (self._maintenance_counters[i] + 1) % self.num_period_per_hours

        return production_rate

    def _call_pilot(self):
        return self.pilot_waiting_time

    def run(self):
        while True:
            # maintenance_factor = 0
            # if random.random() < self.unscheduled_maintenance_prob:
            #     maintenance_factor = self.maintenance_impact_unscheduled
            # factor = 1 - abs(maintenance_factor)
            production = self._generate_production()  # * factor # pour l'instant non pris en compte
            available = self._get_capacity_left()
            to_add = min(production, available)
            if to_add:
                yield self.container.put(to_add)
            self.wasted_production = production - to_add
            yield self.env.timeout(1)

    def pump(self, amount):
        """Transfert de CO2 depuis l'usine vers un navire."""
        pump_rate = self.pump_in_maintenance_rate if self.maintenance else self.pump_rate
        transferred = min(pump_rate, amount, self.container.level)
        if transferred:
            yield self.container.get(transferred)
        return transferred
