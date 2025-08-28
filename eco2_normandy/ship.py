import random
from enum import Enum

from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

import simpy
from simpy import Container
from eco2_normandy.factory import Factory
from eco2_normandy.storage import Storage
from eco2_normandy.logger import Logger
from eco2_normandy.tools import (
    get_distance,
    get_speed,
    wait_for_bad_weather_before_leaving_port,
    get_destination_from_name,
)


WEATHER_LOOK_AHEAD = 10


# --- Enumération pour l'état des navires ---
class shipState(Enum):
    DOCKED = 0  # À quai
    NAVIGATING = 1  # En navigation vers la destination
    WAITING = 2  # En attente (pour quai ou pilote)
    DOCKING = 3  # En cours d'amarrage
    INIT = -1  # État initial

    def __str__(self):
        return self.name


# --- Classe Ship (corrigée pour réintégrer la logique manquante) ---
class Ship(object):
    def __init__(
        self,
        num_period_per_hours: int,
        env: simpy.Environment,
        name: str,
        factory: Factory,
        storages: list,
        weather_station,
        capacity_max: int,
        speed_max: int,
        allowed_speeds: dict,
        distances: dict,
        init: dict,
        staff_cost_per_hour: int,
        usage_cost_per_hour: int,
        immobilization_cost_per_hour: int,
        ship_buying_cost: int,
        fuel_consumption_per_day: float,
        max_percent_capacity: float = 1.0,
        fixed_storage_destination=None,
        logger=None,
        **kwargs,
    ):
        self.logger = logger or Logger()
        self.env = env
        self.name = name
        self.num_period_per_hours = num_period_per_hours
        self.factory = factory
        self.storages = storages
        self.weather_station = weather_station
        self.capacity_max = capacity_max * max_percent_capacity
        self.speed_max = speed_max * num_period_per_hours
        self.allowed_speeds = allowed_speeds
        self.distances = distances
        self.init = init
        self.fixed_storage_destination = fixed_storage_destination

        # Variables d'état
        self.distance_to_go = 0
        self.speed = 0
        self.state: shipState = shipState.INIT
        self.history = []
        self.next_state = None
        self.time_to_wait = 0
        self.time_left_docked = 0
        self.former_destination = None  # Pour l'animation
        self.destination = None

        # Coûts par période
        self.staff_cost_per_period = staff_cost_per_hour * num_period_per_hours
        self.usage_cost_per_period = usage_cost_per_hour * num_period_per_hours
        self.immobilization_cost_per_period = immobilization_cost_per_hour * num_period_per_hours
        self.ship_buying_cost = ship_buying_cost
        self.fuel_consumption_per_day = fuel_consumption_per_day

        capa_init = min(self.init.get("capacity", 0), self.capacity_max)
        self.container = Container(env, init=capa_init, capacity=capacity_max)
        self.dock_req = None  # Permet de tracer les requests faite aux ports
        self.is_docked = False  # Indique si le navire est amarré

        self.__dict__.update(kwargs)
        self.action = env.process(self.run())

    @property
    def capacity(self):
        return self.container.level

    def _save_state(self):
        states_to_save = [
            "speed",
            "distance_to_go",
            "state",
            "destination",
            "capacity",
            "ship_buying_cost",
            "fuel_consumption_per_day",
            "staff_cost_per_period",
            "usage_cost_per_period",
            "immobilization_cost_per_period",
        ]
        self.history.append(
            {k: getattr(self, k) if k != "destination" else getattr(self, k).name for k in states_to_save}
        )

    def _check_if_arrived(self):
        if self.distance_to_go <= 0:
            self.distance_to_go = 0
            if self.state == shipState.NAVIGATING:
                self.state = shipState.DOCKING

    def _navigating(self):
        self.distance_to_go -= self.speed / self.num_period_per_hours
        self._check_if_arrived()

    def _pick_new_destination(self):
        self.former_destination = self.destination  # Pour l'animation
        if isinstance(self.destination, Storage):
            self.distance_to_go = get_distance(self.destination, self.factory, self.distances)
            self.destination = self.factory
        else:
            if self.fixed_storage_destination:
                self.destination = get_destination_from_name(
                    self.fixed_storage_destination, self.factory, self.storages
                )
            else:
                self.destination = random.choice(self.storages)
            self.distance_to_go = get_distance(self.destination, self.factory, self.distances)

    def load_unload(self):
        # Retourne True lorsque le transfert est terminé.
        match self.destination:
            case Storage():
                if self.capacity == 0:
                    return True
                transferred_amount = yield self.env.process(self.destination.pump(self.capacity))
                if transferred_amount:
                    yield self.container.get(transferred_amount)
                return self.capacity == 0

            case Factory():
                free_cap = self.capacity_max - self.capacity
                transferred_amount = yield self.env.process(self.destination.pump(free_cap))
                if transferred_amount:
                    yield self.container.put(transferred_amount)
                return self.capacity == self.capacity_max
        return False

    def get_speed(self, *args, **kwargs):
        # speed = get_speed(*args, **kwargs) # ignorer pour l'instant car le 'weather' n'est pas encore pris en compte
        # speed = min(speed, self.speed_max) # on retourne la vitesse maximale pour que les model puisse l'utiliser
        return self.speed_max

    def run(self):
        # Initialisation à partir des paramètres initiaux
        if self.state == shipState.INIT:
            self.state = shipState[self.init["state"]]
            self.destination = get_destination_from_name(self.init["destination"], self.factory, self.storages)
            if self.state == shipState.DOCKED:
                self.dock_req = self.destination.request()
                self.distance_to_go = self.init["distance_to_go"]
                yield self.dock_req
                self.is_docked = True
                yield self.env.timeout(1)
        while True:
            period = (self.env.now, self.env.now + WEATHER_LOOK_AHEAD)
            weathers = self.weather_station.get_weather_period(*period)
            if self.state == shipState.NAVIGATING:
                self.speed = self.get_speed(weathers[0], self.allowed_speeds)
                self._navigating()
                yield self.env.timeout(1)
            elif self.state == shipState.DOCKING:
                self.dock_req = self.destination.request()
                yield self.dock_req
                self.is_docked = True
                self.state = shipState.DOCKED
                docking_time = (
                    self.destination.lock_waiting_time or 1
                )  # if no lock waiting time (i.e. 0) do a normal step
                yield self.env.timeout(docking_time)
            elif self.state == shipState.DOCKED:
                transfer_finished = yield self.env.process(self.load_unload())
                if transfer_finished and not wait_for_bad_weather_before_leaving_port(weathers, self.allowed_speeds):
                    undock_time = self.destination.lock_waiting_time or 1  # same
                    yield self.env.timeout(undock_time)
                    self.state = shipState.WAITING
                    self.next_state = shipState.NAVIGATING
                    self.time_to_wait = self.destination.transit_time_from_dock
                    self.destination.release(self.dock_req)
                    self.dock_req = None
                    self.is_docked = False
                else:
                    yield self.env.timeout(1)
            elif self.state == shipState.WAITING:
                if self.time_to_wait == 0:
                    self.state = shipState.DOCKING if self.next_state is None else self.next_state
                    if self.state == shipState.NAVIGATING:  # On quitte le port
                        self._pick_new_destination()
                        self.speed = self.get_speed(weathers[0], self.allowed_speeds)
                    else:
                        self.distance_to_go = self.destination._get_pbs_to_dock()
                    self.next_state = None
                    self._navigating()
                else:
                    yield self.env.timeout(self.time_to_wait)
                    self.time_to_wait = 0
