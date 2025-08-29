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
from eco2_normandy.tools import get_simlulation_variable
from optimizer.boundaries import ConfigBoundaries
from optimizer.utils import ConfigBuilderFromSolution

logger = Logger()
boundaries = ConfigBoundaries()
cfg_builder = ConfigBuilderFromSolution(None, boundaries)
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
        "num_period": st.number_input("Number of Periods", value=2000),
        "distances": {
            "Le Havre": {
                "Rotterdam": 263.0,
                "Bergen": 739.0,
            }
        },
    }


with st.expander("Factory"):
    factory = {
        "number_of_tanks": st.number_input("Number of Tanks", value=1, min_value=1),
        "num_sources": st.number_input("Number of Sources", value=5, min_value=1),
    }
    annual_production_capacity = st.number_input("Annual Production Capacity (all sources)", value=570_000)

num_source = factory["num_sources"]
factory["sources"] = [ { "name": f"Source {i+1}", 
                        "annual_production_capacity": annual_production_capacity//num_source,
                        "maintenance_rate":0.1,
                        } for i in range(num_source) ]

with st.expander("Storages"):
    storage = {
        "num_storages": st.number_input("Number of Storages", value=1, min_value=1, max_value=2),
        "storage_caps": st.number_input("Total Storage Capacity (m3)", value=10_000, min_value=10_000),
        "use_Bergen": st.checkbox("Use Bergen as storage location", value=False),
        'use_Rotterdam': st.checkbox("Use Rotterdam as storage location", value=True)
    }

map_ship_initial_destination = {"Le Havre": 0, "Rotterdam": 1, "Bergen": 2}
map_ship_fixed_storage_destination = {"Rotterdam": 0, "Bergen": 1}
with st.expander("Ships"):
    ships = {
        "num_ship": st.number_input("Number of Ships", value=1, min_value=1),
        "ship_capacity": st.number_input("Ship Capacity (m3)", value=12_000, min_value=1_000),
        "ship_speed": st.number_input("Ship Speed (knots)", value=12, min_value=1),
    }
    for i in range(ships["num_ship"]):
        st.write(f"Ship {i + 1}")
        ships = {
            f"init{i+1}_destination": map_ship_initial_destination[st.selectbox(f"Select {i+1} Destination", options=["Le Havre", "Rotterdam", "Bergen"])],
            f"fixed{i+1}_storage_destination": map_ship_fixed_storage_destination[st.selectbox(f"Select {i+1} Fixed Storage Destination", options=["Rotterdam", "Bergen"])],
            **ships
        }


config = get_simlulation_variable("scenarios/dev/phase3_bergen_18k_2boats.yaml")[0]
config["general"].update(general)
config["factory"].update(factory)


solution = {**factory, **storage, **ships}
cfg_builder.base_config = config
sim_config = cfg_builder.build(solution)
simulation = Simulation(config=sim_config)

# Validation before simulation
if st.button("Simulate", disabled=st.session_state.get("simulation_state", None) == "running"):
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
    process = multiprocessing.Process(target=launch_pygame_animation, args=(sim_config,))
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
        config=sim_config,
    )

    plots = []
    plots.append(generator.plot_factory_capacity_evolution())
    plots.append(generator.plot_storage_capacity_comparison())
    plots.append(generator.plot_factory_wasted_production_over_time())
    plots.append(generator.plot_travel_duration_evolution())
    plots.append(generator.plot_waiting_time_evolution())
    plots.append(generator.plot_co2_transportation())
    plots.append(generator.plot_cost_kpis_table())
    plots.append(generator.plot_metric_kpis_table())

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
