import time
import os
import multiprocessing
import logging

import streamlit as st

from GUI import PGAnime
from eco2_normandy import Simulation
from eco2_normandy import Simulation
from KPIS import Kpis
from eco2_normandy.logger import Logger

logger = Logger()
logging.basicConfig(level=logging.ERROR)
# Optional: Set the page to use the entire width
st.set_page_config(layout="wide")
st.logo(image=os.path.join("assets", "logo.png"), link="https://tnpconsultants.com/")
st.markdown(
    """
<style>
div[class*="factories"],
div[class*="storages"],
div[class*="ships"] {
    /* Card style */
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    background-color: #fff;
    transition: transform 0.3s ease;
}

div[class*="factories"]:hover,
div[class*="storages"]:hover,
div[class*="ships"]:hover {
    transform: translateY(-5px);
}

</style>
""",
    unsafe_allow_html=True,
)


# Streamlit App
st.title("Simple Simulation Input and KPI Graphs Generator")

st.session_state.simulation_name = st.text_input("Nom de la simulation", value="simulation 1")

with st.expander("General Inputs"):
    general = {
        "num_period_per_hours": st.number_input("Number of Periods per Hour", value=1.0),
        "num_ships": st.number_input("Number of Ships", value=1),
        "num_period": st.number_input("Number of Periods", value=2000),
        "distances": {
            "Le Havre": {
                "Rotterdam": 263.0,
                "Bergen": 739.0,
            }
        },
    }


weather_probability = {
    "wind": 0.1,
    "waves": 0.1,
    "current": 0.1,
}

kpis = {
    "fuel_price_per_ton": 520,
    "delay_penalty_per_hour": 200,
    "co2_release_cost_per_ton": 0.1,
    "storage_cost_per_m3": 30,
}

allowed_speeds = {
    "wind": {"6": 12, "10": 10, "20": 0},
    "wave": {"6": 12, "10": 10, "20": 0},
    "current": {"6": 12, "10": 10, "20": 0},
}


# Manage dynamic items
def manage_dynamic_section(section_name, section_state_key, default_item):
    with st.expander(section_name):
        if section_state_key not in st.session_state:
            st.session_state[section_state_key] = [default_item.copy()]

        for idx, item in enumerate(st.session_state[section_state_key]):
            with st.container(key=f"{section_state_key}_{idx}"):
                st.subheader(f"{section_name[:-1]} {idx + 1}")
                col1, col2 = st.columns([5, 1])

                with col1:
                    for key, value in item.items():
                        if isinstance(value, (float, int)):  # Handle numeric inputs
                            st.session_state[section_state_key][idx][key] = st.number_input(
                                f"{section_name[:-1]} {idx + 1} - {key.capitalize()}",
                                value=value,
                            )
                        elif isinstance(value, str):  # Handle string inputs
                            st.session_state[section_state_key][idx][key] = st.text_input(
                                f"{section_name[:-1]} {idx + 1} - {key.capitalize()}",
                                value=value,
                            )
                        elif isinstance(value, dict):  # Handle nested dictionaries
                            with st.container():
                                st.subheader(f"{key.capitalize()} Details")
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, (float, int)):  # Numeric sub-values
                                        st.session_state[section_state_key][idx][key][sub_key] = st.number_input(
                                            f"{section_name[:-1]} {idx + 1} - {key.capitalize()} - {sub_key.capitalize()}",
                                            value=sub_value,
                                        )
                                    elif isinstance(sub_value, str):  # String sub-values
                                        st.session_state[section_state_key][idx][key][sub_key] = st.text_input(
                                            f"{section_name[:-1]} {idx + 1} - {key.capitalize()} - {sub_key.capitalize()}",
                                            value=sub_value,
                                        )

                with col2:
                    if len(st.session_state[section_state_key]) > 1 and st.button(
                        f"❌", key=f"{section_name}_delete_{idx}"
                    ):
                        st.session_state[section_state_key].pop(idx)
                        st.rerun()

        if st.button(f"Add {section_name[:-1]}"):
            st.session_state[section_state_key].append(default_item.copy())
            st.rerun()


# Factories
manage_dynamic_section(
    "Factory",
    "factory",
    {
        "name": "Le Havre",
        "capacity_max": 10000.0,
        "annual_production_capacity": 530000,
        "production_rate": 60.5,
        "maintenance_rate": 0.2,
        "docks": 1,
        "number_of_tanks": 4,
        "cost_per_tank": 3500,
        "pbs_to_dock": 2,
        "scheduled_maintenance_period_major": 3,
        "scheduled_maintenance_period_minor": 1.5,
        "unscheduled_maintenance_prob": 0.01,
        "pump_rate": 1000,
        "pump_in_maintenance_rate": 500,
        "loading_time": 3,
        "weather_waiting_time": 1,
        "pilot_waiting_time": 2,
        "dock_waiting_time": 3.5,
        "lock_waiting_time": 2,
        "transit_time_to_dock": 5.3,
        "transit_time_from_dock": 5.3,
        "storage_cost_per_m3": 35,
        "initial_capacity": 1,
        "sources": [
            {
                "name": "source 1",
                "annual_production_capacity": 600000,
                "maintenance_rate": 0.1,
            }
        ],
    },
)

# Storages
manage_dynamic_section(
    "Storages",
    "storages",
    {
        "name": "Rotterdam",
        "capacity_max": 1200,
        "cost_per_tank": 5000,
        "number_of_tanks": 1800000000,
        "consumption_rate": 1000,
        "maintenance_rate": 0.2,
        "docks": 1,
        "pbs_to_dock": 2,
        "pump_rate": 10000,
        "pump_in_maintenance_rate": 5000,
        "lock_waiting_time": 0,
        "unloading_time": 3,
        "transit_time_to_dock": 3.3,
        "transit_time_from_dock": 3.3,
        "storage_cost_per_m3": 35,
    },
)

# Ships
manage_dynamic_section(
    "Ships",
    "ships",
    {
        "name": "Ship 1",
        "capacity_max": 1200.0,
        "speed_max": 15.0,
        "init": {
            "state": "DOCKED",
            "destination": "Le Havre",
            "distance_to_go": 0,
            "capacity": 0,
        },
        "immobilization_cost_per_hour": 45.0,
        "staff_cost_per_hour": 130.0,
        "usage_cost_per_hour": 300.0,
        "ship_buying_cost": 1500000,
        "fuel_consumption_per_day": 20,
        "fixed_storage_destination": "Rotterdam",
    },
)


# Combine all inputs
simulation_variables = {
    "general": general,
    "weather_probability": weather_probability,
    "KPIS": kpis,
    "factory": st.session_state.factory[0],
    "storages": st.session_state.storages,
    "ships": st.session_state.ships,
    "allowed_speeds": allowed_speeds,
}

simulation = Simulation(config=simulation_variables)

# Validation before simulation
if st.button("Simulate", disabled=st.session_state.get("simulation_state", None) == "running"):
    # Check if conditions are met before running the simulation
    if len(st.session_state.factory) < 1:
        st.error("You must have one factory.")
    elif len(st.session_state.storages) < 1:
        st.error("You must have at least one storage.")
    elif len(st.session_state.ships) < 1:
        st.error("You must have at least one ship.")
    else:
        st.session_state.plots = []

        st.session_state.started_at = time.time()
        st.session_state.done_at = None
        st.session_state.simulation_state = "running"
        st.rerun()


def launch_pygame_animation(config):
    return PGAnime(config).run()


if st.button(
    "PyGame Animation 🚀",
    disabled=st.session_state.get("pygame_running", "") == "running",
):
    st.session_state.pygame_running = "running"
    process = multiprocessing.Process(target=launch_pygame_animation, args=(simulation_variables,))
    process.start()
    st.success("Animation lancer !")
    st.session_state.pygame_process = process
    st.rerun()

if st.button(
    "Arrêter l'animation 🛑",
    disabled=st.session_state.get("pygame_running", "") != "running",
):
    if "pygame_process" in st.session_state:
        st.session_state.pygame_process.terminate()
        del st.session_state["pygame_process"]
        st.session_state.pygame_running = None
        st.success("Animation arrêtée !")
        st.rerun()

if st.session_state.get("pygame_running") == "running":
    proc = st.session_state.get("pygame_process")
    # If the process object exists but is no longer alive…
    if not proc.is_alive():
        # clean up
        proc.join()  # reap any zombie
        del st.session_state["pygame_process"]
        st.session_state.pygame_running = None
        st.success("🎮 Pygame window was closed — animation stopped.")

# Debut de la simulation
if st.session_state.get("simulation_state", None) == "running":
    st.info("Simulation started...")
    simulation.run()
    generator = Kpis(
        simulation.result,
        config=simulation_variables,
    )

    plots = []
    plots.append(generator.plot_factory_capacity_evolution())
    plots.append(generator.plot_storage_capacity_comparison())
    plots.append(generator.plot_factory_wasted_production_over_time())
    plots.append(generator.plot_travel_duration_evolution())
    plots.append(generator.plot_waiting_time_evolution())
    plots.append(generator.plot_co2_transportation())
    plots.append(generator.plot_cost_kpis_table())
    plots.append(generator.plot_metric_kpis_table()) #todo: fix error

    st.session_state.plots = plots
    st.session_state.simulation_state = "done"
    st.session_state.done_at = time.time()
    st.rerun()


# Fin de la simultion
if len(st.session_state.get("plots", [])) > 0 and st.session_state.get("simulation_state", None) == "done":
    st.success(f"Simulation completed in {round(st.session_state.done_at - st.session_state.started_at)} seconds.")
    # Create two columns
    col1, col2 = st.columns(2)
    plots = st.session_state.plots

    # Iterate over the plots and display them in columns
    for i, plot in enumerate(plots):
        if i % 2 == 0:
            with col1:
                st.plotly_chart(plot)
        else:
            with col2:
                st.plotly_chart(plot)
