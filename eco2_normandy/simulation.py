import simpy
import time
from colorama import Fore

from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

from eco2_normandy.factory import Factory
from eco2_normandy.storage import Storage
from eco2_normandy.ship import Ship
from eco2_normandy.weather import WeatherStation
from eco2_normandy.stateSaver import StateSaver
from eco2_normandy.logger import Logger
from eco2_normandy.tools import data_to_dataframe
from KPIS import LiveKpisGraphsGenerator

# --- Classe Simulation (corrigée pour réintégrer collecte de données et KPI) ---
class Simulation:
    n_simu = 0
    def __init__(self, config_name=None, config=None, data_df=None, kpis=None, generator=None, logger=None, verbose=True):
        Simulation.n_simu += 1
        self.logger = logger or Logger()
        self.config_name = config_name
        self.config = config
        self.data_df = data_df
        self.kpis = kpis
        self.generator = generator
        self.verbose = verbose
        self._init_simulation()

    def _init_simulation(self):
        self.env = simpy.Environment()
        self.NUM_PERIOD = self.config["general"]["num_period"]
        self.NUM_PERIOD_IN_HOURS = self.config["general"]["num_period_per_hours"]

        # Initialisation de la station météo (exemple simplifié)
        weather_probability = self.config.get("weather_probability", {})
        values = {
            "wind": [int(k) for k, _ in self.config.get("allowed_speeds").get("wind").items()],
            "wave": [int(k) for k, _ in self.config.get("allowed_speeds").get("wave").items()],
            "current": [int(k) for k, _ in self.config.get("allowed_speeds").get("current").items()],
        }
        self.weather_station = WeatherStation(
            values=values["wind"],
            num_period=self.NUM_PERIOD,
            weather_probability=weather_probability,
            logger = self.logger
        )

        # Création de l'usine (Factory)
        self.factory = Factory(
            num_period_per_hours=self.NUM_PERIOD_IN_HOURS, env=self.env, logger=self.logger, **self.config["factory"]
        )

        # Création des Storage
        self.storages = [
            Storage(num_period_per_hours=self.NUM_PERIOD_IN_HOURS, 
                    env=self.env, logger=self.logger, **storage)
            for storage in self.config["storages"]
        ]

        # Création des navires (Ships)
        self.ships = [
            Ship(
                num_period_per_hours=self.NUM_PERIOD_IN_HOURS,
                env=self.env,
                factory=self.factory,
                storages=self.storages,
                weather_station=self.weather_station,
                distances=self.config["general"]["distances"],
                allowed_speeds=self.config["allowed_speeds"],
                logger=self.logger,
                **ship,
            )
            for ship in self.config["ships"]
        ]
        StateSaver(self.env, self.factory, self.storages, self.ships)

    def run(self):
        if self.verbose: self.logger.info(Fore.CYAN + f"Starting simulation n°{Simulation.n_simu}..."+Fore.RESET)
        start_time = time.perf_counter()
        self.env.run(until=self.NUM_PERIOD)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        if self.verbose:
            self.logger.info(Fore.GREEN + "Simulation done, collecting data..." + Fore.RESET)
            self.logger.info(Fore.GREEN + f"⏱️ Simulation time: {elapsed:.6f} seconds" + Fore.RESET)

    def step(self):
        if self.env.peek() < self.NUM_PERIOD:
            self.env.step()

    @property
    def result(self):
        return data_to_dataframe(
            storages=self.storages, factory=self.factory, ships=self.ships
        )

    def get_kpis_generator(self):
        if self.generator:
            self.generator.upload_data(self.result)
            return self.generator
        return LiveKpisGraphsGenerator(
            self.result,
            config=self.config,
        )

    def generate_kpis(self):
        if self.generator is None:
            self.get_kpis_generator()
        # self.generator._trip_analysis()
        self.kpis = self.generator.generate_kpis_graphs()


if __name__ == "__main__":
    from eco2_normandy.tools import get_simlulation_variable
    path = "scenarios\dev\phase3_bergen_18k_2boats.yaml"
    path = "config.yaml"
    config = get_simlulation_variable(path)[0]
    config["general"]["num_period"] = 1000
    sim = Simulation("test", config, verbose=True)
    sim.run()
