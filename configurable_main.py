import os
import itertools
import yaml
import logging
import argparse
import numpy as np  # For handling ranges
import colorama
from colorama import Fore, Style

from KPIS.KpisGraphsGenerator import KpisGraphsGenerator
from eco2_normandy.simulation import Simulation
from eco2_normandy.tools import get_simlulation_variable

# Initialize colorama
colorama.init(autoreset=True)

logger = logging.getLogger()
logging.basicConfig(level=logging.WARNING)



def process_config_file(file_path) -> list[Simulation]:

    print(Fore.GREEN + f"Using config file: {file_path}")
    # config_name = os.path.basename(file_path).replace(".yaml", "").replace("/","_")
    config_name = file_path.replace(".yaml", "").replace("/", "_")
    
    # Generate all combinations of parameters
    param_combinations = get_simlulation_variable(file_path)
    simulations = []

    print(Fore.GREEN + "Starting the simulation journey... Hold tight!")

    for i, config in enumerate(param_combinations):
        # Run only one simulation
        logger.info(Fore.MAGENTA + "Generating KPIs for current configuration...")
        simulation = Simulation(config_name=config_name, config=config)
        simulation.run()
        print(Fore.GREEN + "Simulation done.")
        logger.info("Generating KPIS...")
        simulation.generate_kpis()
        logger.info(Fore.GREEN + "KPIs generated successfully.")
        simulations.append(simulation)

    return simulations


def collect_files(path) -> tuple[str, str]:
    all_files = []

    if os.path.isfile(path):  # If one single config file
        if path.endswith(".yaml"):
            all_files.append(path)

    if os.path.isdir(path):  # If multiple config files
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".yaml"):
                    all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    # Display project information and copyright
    project_name = "ECO2 Normandy Simulation Framework"
    copyright_notice = "© 2025 TNP Consultants and AirLiquide. All Rights Reserved."
    intro_message = f"""
    {Fore.CYAN}{Style.BRIGHT}***************************************
    {Fore.GREEN}Welcome to the {project_name}!
    {Fore.YELLOW}{copyright_notice}
    {Fore.CYAN}***************************************
    """

    print(intro_message)
    print(Fore.BLUE + "Initializing simulation environment...")

    # Load parameter ranges from YAML

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    arg = parser.parse_args().config

    files_list = collect_files(arg)
    output_folder_name = ""

    simulations = []
    for file in files_list:
        if file.endswith(".yaml"):
            simulation = process_config_file(file)
            simulations.extend(simulation)

    output_folder_name = os.path.join(
        "KPIs", "generated_graphs", "le_havre_capacity_evolution_comparison"
    )

    KpisGraphsGenerator.generate_html_page_with_plots(
        [
            KpisGraphsGenerator.plot_factories_capacity_evolution(simulations),
            KpisGraphsGenerator.plot_summary_table_for_each_scenario(simulations),
            KpisGraphsGenerator.plot_summary_costs_for_each_scenario(simulations),
        ],
        output_folder_name,
        full_page=True,
    )

    print(Fore.GREEN + "Thank you for using the ECO2 Normandy Simulation Framework!")
