import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import itertools
import yaml
import logging

from .factory import Factory
from .storage import Storage
from .weather import WeatherReport
from .logger import Logger

output_folder = "system_states"
date_format = "%Y%m%d%H%M%S"
date = datetime.now().strftime(date_format)
filename_format = "{output_folder}/{config_name}_{date}.csv"


def data_to_dataframe(
    factory: Factory, storages: list[Storage], ships: list[object]
) -> pd.DataFrame:
    entities = [factory] + storages + ships
    dfs = []
    for ent in entities:
        # transforme la liste de dicts en DataFrame, indexée par step
        df_ent = pd.DataFrame(ent.history)
        df_ent.index.name = "step"
        # crée un MultiIndex (entité -> attribut)
        df_ent.columns = pd.MultiIndex.from_product([[ent.name], df_ent.columns])
        dfs.append(df_ent)
    # concatène tout d’un coup sur les colonnes
    return pd.concat(dfs, axis=1)

# old dataframe creation, work with this version of kpis generator 
# need refactor to use new dataframe creation
def former_data_to_dataframe(
    factory: Factory, storages: list[Storage], ships: list[object]
) -> pd.DataFrame:
    data = {
        factory.name: factory.history,
    }
    storages = {s.name: s.history for s in storages}
    data.update(storages)
    ships = {s.name: s.history for s in ships}
    data.update(ships)
    return pd.DataFrame(data)


def save_dataframe_to_csv(df: pd.DataFrame, config_name: str):
    filename = filename_format.format(
        output_folder=output_folder, date=date, config_name=config_name
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, sep=";")
    return filename


def get_distance(storage: Storage, factory: Factory, distances: dict):
    return distances[storage.name][factory.name]


def get_speed(weather: WeatherReport, speed_dict: dict, catchup=False) -> int:
    speed_current = max(
        [v for _, v in speed_dict["current"].items()]
    )  # speed dictated by the current

    speed_wind = max(
        [v for _, v in speed_dict["wind"].items()]
    )  # speed dictated by the wind

    return min(speed_current, speed_wind)


def wait_for_bad_weather_before_leaving_port(
    weathers: list[WeatherReport], speed_dict
) -> bool:
    max_allowed_current = max([k for k, _ in speed_dict["current"].items()])
    max_allowed_wind = max([k for k, _ in speed_dict["wind"].items()])

    return max_allowed_current in [v.current for v in weathers] or max_allowed_wind in [
        v.wind for v in weathers
    ]


def get_destination_from_name(
    name: str, factory: Factory, storages: list[Storage]
) -> Storage | Factory :
    if factory.name == name:
        return factory
    for s in storages:
        if s.name == name:
            return s
    raise ValueError(f"Destination {name} not found in factory or storages.")

def generate_combinations(parameter_ranges) -> list:
    if isinstance(parameter_ranges, dict) and "range" in parameter_ranges:
        # Handle range specifications
        start, stop, step = parameter_ranges["range"]
        return list(
            np.arange(start, stop + step, step)
        )  # Include stop value in the range

    elif isinstance(parameter_ranges, dict):
        # Recursively handle nested dictionaries
        keys, values = zip(
            *[
                (key, generate_combinations(value))
                for key, value in parameter_ranges.items()
            ]
        )
        return [
            dict(zip(keys, combination)) for combination in itertools.product(*values)
        ]
    elif isinstance(parameter_ranges, list) and all(
        isinstance(item, dict) for item in parameter_ranges
    ):
        # Handle lists of dictionaries (e.g., multiple storages or ships)
        combinations = [generate_combinations(item) for item in parameter_ranges]
        return [list(comb) for comb in itertools.product(*combinations)]
    elif isinstance(parameter_ranges, list):
        # If it's a simple list of values, return it as-is
        return parameter_ranges
    else:
        # For scalar values, wrap in a list for consistency
        return [parameter_ranges]

def get_simlulation_variable(file_path):
    path = (Path.cwd() / file_path ).resolve()
    with open(str(path), "r") as file:
        parameter_ranges = yaml.safe_load(file)
    return generate_combinations(parameter_ranges)

    
def get_Logger():
    """
    Crée un logger configuré pour l'application.
    Le logger enregistre les messages dans la console et dans un fichier.
    """
    return Logger()
