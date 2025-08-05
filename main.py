import simpy
import yaml
import logging
from eco2_normandy.factory import Factory
from eco2_normandy.storage import Storage
from eco2_normandy.ship import Ship
from eco2_normandy.weather import WeatherStation, WeatherReport
from eco2_normandy.tools import data_to_dataframe, save_dataframe_to_csv

from eco2_normandy.stateSaver import StateSaver
from KPIS.KpisGraphsGenerator import KpisGraphsGenerator

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

NUM_PERIOD_IN_HOURS = 1
# Load the YAML configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

NUM_SHIPS = 2
NUM_PERIOD = 24
DISTANCES = {"factory": {"storage": 100}}

WEATHER_PROBABILITY = {"wind": 0.1, "waves": 0.1, "current": 0.1}

ships_init = [
    {"state": "DOCKED", "destination": "storage", "distance_to_go": 6000},
    {"state": "DOCKED", "destination": "factory", "distance_to_go": 0, "capacity": 0},
]

output_folder = "output"
allowed_speeds = {
    "wind": {
        "6": 12 * NUM_PERIOD_IN_HOURS,
        "10": 10 * NUM_PERIOD_IN_HOURS,
        "20": 0 * NUM_PERIOD_IN_HOURS,
    },
    "current": {
        "6": 12 * NUM_PERIOD_IN_HOURS,
        "10": 10 * NUM_PERIOD_IN_HOURS,
        "20": 0 * NUM_PERIOD_IN_HOURS,
    },
}


def _generate_weather_report():
    return WeatherReport(wind=0, waves=0, current=0)


def weather_generator() -> list[WeatherReport]:
    return [_generate_weather_report() for _ in range(NUM_PERIOD)]


if __name__ == "__main__":
    env = simpy.Environment()

    weather_station = WeatherStation(generator=weather_generator)

    factory = Factory(
        env,
        "factory",
        capacity_max=26000,
        production_rate=3000,
        maintenance_rate=0.2,
        docks=1,
        number_of_tanks=4,
        cost_per_tank=3500,
        pbs_to_dock=2,
        annual_production_capacity=999999999,
        scheduled_maintenance_period_major=3,
        scheduled_maintenance_period_minor=1.5,
        unscheduled_maintenance_prob=0.01,
        pump_rate=1000 * NUM_PERIOD_IN_HOURS,
        pump_in_maintenance_rate=500 * NUM_PERIOD_IN_HOURS,
        loading_time=3 / NUM_PERIOD_IN_HOURS,
        weather_waiting_time=1 / NUM_PERIOD_IN_HOURS,
        pilot_waiting_time=2 / NUM_PERIOD_IN_HOURS,
        dock_waiting_time=3.5 / NUM_PERIOD_IN_HOURS,
        lock_waiting_time=2 / NUM_PERIOD_IN_HOURS,
        transit_time_to_dock=5.3 / NUM_PERIOD_IN_HOURS,
        transit_time_from_dock=5.3 / NUM_PERIOD_IN_HOURS,
        storage_cost_per_m3=35,
    )

    storages = [
        Storage(
            env,
            "storage",
            capacity_max=9999999999999999999,
            cost_per_tank=5000,
            number_of_tanks=1800000000,
            consumption_rate=1000,
            maintenance_rate=0.2,
            docks=1,
            pbs_to_dock=2,
            pump_rate=1000 * NUM_PERIOD_IN_HOURS,
            pump_in_maintenance_rate=500 * NUM_PERIOD_IN_HOURS,
            lock_waiting_time=0 * NUM_PERIOD_IN_HOURS,
            unloading_time=3 * NUM_PERIOD_IN_HOURS,
            transit_time_to_dock=3.3 / NUM_PERIOD_IN_HOURS,
            transit_time_from_dock=3.3 / NUM_PERIOD_IN_HOURS,
            storage_cost_per_m3=35,
        )
    ]

    ships = [
        Ship(
            env,
            name=f"ship{i+1}",
            factory=factory,
            storages=storages,
            weather_station=weather_station,
            capacity_max=5000,
            speed_max=15,
            allowed_speeds=allowed_speeds,
            distances=DISTANCES,
            init=ships_init[i],
            immobilization_cost_per_hour=45,
            staff_cost_per_hour=130,
            usage_cost_per_hour=300,
            ship_buying_cost=1500000,
            fuel_consumption_per_day=20,
        )
        for i in range(NUM_SHIPS)
    ]

    stateSaver = StateSaver(env, factory, storages, ships)

    logger.info("Starting simulation")
    env.run(until=NUM_PERIOD)

    logger.info("Simulation done, collecting data...")

    data_df = data_to_dataframe(storages=storages, factory=factory, ships=ships)

    logger.info("Wrtiting results...")
    save_dataframe_to_csv(data_df)

    logger.info("Generating KPIS...")
    generator = KpisGraphsGenerator(data_df)
    generator.generate_kpis_graphs()

    logger.info("Done.")
