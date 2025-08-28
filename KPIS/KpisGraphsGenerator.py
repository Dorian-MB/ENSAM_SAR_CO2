import os
import sys
import ast
import shutil
from colorama import Fore, Style
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from eco2_normandy.tools import former_data_to_dataframe as data_to_dataframe

pio.templates.default = "ggplot2"


class KpisGraphsGenerator:
    generated_graphs_index = 0
    generated_excel_files_index = 0
    graphs_data_dictionnary = {}
    travel_data = None
    waiting_data = None
    trips = None
    initial_investment = 0
    functional_costs = 0
    combined_cost = 0

    def __init__(
        self,
        simulation_df,
        config,
        storage_columns=["storage"],
        factory_column="factory",
        show_tables=False,
        ouput_folder="generated_graphs",
        output_excels=True,
    ):
        self.show_tables = show_tables
        self.config = config
        self.output_excels = output_excels
        self.df = simulation_df
        self.output_folder = os.path.join(
            "KPIS",
            "generated_graphs",
            ouput_folder,
        )
        self.ship_names = [s["name"] for s in self.config["ships"]]
        try:
            shutil.rmtree(self.output_folder)
        except Exception as err:
            print("Folder doesn't exist, creating it: ", self.output_folder)

        os.makedirs(self.output_folder, exist_ok=True)
        # rename first column to indicate steps/phases of simulation
        self.df.index.names = ["step"]
        self.df = self.df.reset_index()
        # Dynamically identify ship columns
        self.factory_column = self.config["factory"]["name"]
        self.storage_columns = [s["name"] for s in self.config["storages"]]
        self.ship_columns = self.df.columns[
            len(storage_columns) + 2 :
        ].tolist()  # All columns after 'storage'

        self.kpis = config["KPIS"]
        # self._trip_analysis()

    def _trip_analysis(self):
        ships = [s["name"] for s in self.config["ships"]]
        self.trips = {s: [] for s in ships}
        factory_name = self.config["factory"]["name"]
        storage_names = [s["name"] for s in self.config["storages"]]
        for ship in ships:
            data = self.df[ship]
            init_new_trip = False
            trip = {
                "DOCKED": 0,
                "DOCKING": 0,
                "WAITING": 0,
                "NAVIGATING": 0,
                "LOADING": 0,
                "UNLOADING": 0,
            }

            for _, l in data.items():
                if init_new_trip:
                    trip = {
                        "DOCKED": 0,
                        "DOCKING": 0,
                        "WAITING": 0,
                        "NAVIGATING": 0,
                        "LOADING": 0,
                        "UNLOADING": 0,
                    }
                    init_new_trip = False

                trip[l.get("state")] += 1
                if l.get("destination") == factory_name and l.get("state") == "DOCKED":
                    trip["LOADING"] += 1
                elif (
                    l.get("destination") in storage_names and l.get("state") == "DOCKED"
                ):
                    trip["UNLOADING"] += 1

                if (
                    l.get("state") == "DOCKED"
                    and l.get("destination") == factory_name
                    and l.get("capacity") in ["0", "0.0"]
                ):
                    init_new_trip = True
                    self.trips[ship].append(trip)

    @staticmethod
    def _format_costs(val):
        return "{:,.2f} €".format(round(val, 2))

    @staticmethod
    def _format_time(val):
        return "{:d} H".format(round(val))

    @staticmethod
    def _format_quantity(val):
        return "{:d} Tons".format(round(val))

    def save_plt_to_image(self, fig_name):

        self.generated_graphs_index += 1
        plt.savefig(
            os.path.join(
                self.output_folder, f"{self.generated_graphs_index} {fig_name}"
            )
        )

    def save_sheets_in_memory(self, dataframes, sheet_name):
        """
        Stores a list of DataFrames and their corresponding sheet names in memory.

        :param dataframes: A list of DataFrame objects.
        :param sheet_name: The name of the sheet to add the DataFrame to.
        """
        if not self.output_excels:
            return
        self.generated_excel_files_index += 1
        if not hasattr(self, "sheets_memory"):
            self.sheets_memory = []  # Initialize the sheets memory if it doesn't exist

        for df in dataframes:
            self.sheets_memory.append({"dataframe": df, "sheet_name": sheet_name})
        print(f"Sheets stored in memory: {sheet_name}")

    def save_all_sheets_to_excel(self):
        """
        Saves all stored DataFrames in memory to an Excel file.
        The Excel file is named "Graphs data.xlsx" by default.

        :param mode: Whether to append to an existing Excel file (True) or create a new one (False).
        """
        excel_file = "Graphs data.xlsx"  # Default file name
        output_path = os.path.join(self.output_folder, excel_file)

        # Check if the sheets_memory is empty
        if not hasattr(self, "sheets_memory") or not self.sheets_memory:
            print("No sheets to save.")
            return

        # Save to Excel
        if os.path.exists(output_path):
            # Open the existing Excel file in append mode
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
                for sheet in self.sheets_memory:
                    sheet["dataframe"].to_excel(
                        writer, sheet_name=sheet["sheet_name"], index=False
                    )
        else:
            # Create a new Excel file
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                for sheet in self.sheets_memory:
                    sheet["dataframe"].to_excel(
                        writer, sheet_name=sheet["sheet_name"], index=False
                    )

        # Clear the sheets_memory after saving
        self.sheets_memory.clear()
        print(f"All sheets saved to Excel file: {output_path}")

    def save_dataframe_to_excel(self, dataframes, sheet_name, mode=False):
        """
        Saves a list of DataFrames to an Excel file, appending each DataFrame as a new sheet.
        The Excel file is named "Graphs data.xlsx" by default.

        :param dataframes: A list of DataFrame objects.
        :param sheet_name: The name of the sheet to add the DataFrame to.
        """
        if not self.output_excels:
            return
        self.generated_excel_files_index += 1
        if mode:
            excel_file = "Graphs data.xlsx"  # Default file name
            output_path = os.path.join(self.output_folder, f"{excel_file}")

            # Check if the file already exists
            if os.path.exists(output_path):
                # Open the existing Excel file in append mode
                with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
                    # Iterate over the dataframes and save each one to a new sheet
                    for i, df in enumerate(dataframes):
                        sheet_name_with_index = f"{sheet_name}"
                        df.to_excel(
                            writer, sheet_name=sheet_name_with_index, index=False
                        )

            else:
                # If the file doesn't exist, create a new one
                with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                    for i, df in enumerate(dataframes):
                        sheet_name_with_index = f"{sheet_name}"
                        df.to_excel(
                            writer, sheet_name=sheet_name_with_index, index=False
                        )

            print(f"Sheet added to Excel File: {sheet_name}")

    @staticmethod
    def plot_Sensitivity_Analysis_of_Costs_Based_on_Storage_Tank_Size(
        storage_sizes, operational_costs_storage
    ):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            storage_sizes,
            operational_costs_storage,
            color="blue",
            label="Storage Sizes",
        )
        plt.plot(storage_sizes, operational_costs_storage, linestyle="--", color="blue")
        plt.title("Sensitivity Analysis of Costs Based on Storage Tank Size")
        plt.xlabel("Storage tank size (m³)")
        plt.ylabel("Functional costs (€)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_factory_capacity_evolution(self, return_html=True):
        title = f"{self.factory_column} Capacity"

        # Extract storage capacity
        self.df["storage_capacity"] = self.df[self.factory_column].apply(
            lambda x: x["capacity"] if x else None
        )

        # Calculate the median storage capacity
        median_capacity = self.df["storage_capacity"].median()

        # Extract capacity_max for percentage calculation and plotting
        capacity_max = (
            self.df[self.factory_column].iloc[0].get("capacity_max", 1)
        )  # Avoid division by zero
        percentage = (median_capacity / capacity_max) * 100 if capacity_max > 0 else 0

        # Prepare data for saving to Excel
        factory_capacity_data = pd.DataFrame(
            {"Step": self.df["step"], title: self.df["storage_capacity"]}
        )

        # Save the data to Excel
        self.save_sheets_in_memory([factory_capacity_data], title)
        self.graphs_data_dictionnary[f"{title} Evolution"] = (
            factory_capacity_data.to_dict(orient="records")
        )

        # Create a Plotly figure
        fig = go.Figure()

        # Add data to the plot
        fig.add_trace(
            go.Scatter(
                x=self.df["step"],
                y=self.df["storage_capacity"],
                mode="lines+markers",
                name=title,
                visible=True,
            )
        )
        # Add a horizontal line for the maximum capacity
        fig.add_trace(
            go.Scatter(
                x=[
                    self.df["step"].min(),
                    self.df["step"].max(),
                ],  # Cover the full x-axis range
                y=[capacity_max, capacity_max],  # Fixed y-value for the horizontal line
                mode="lines",
                line=dict(color="black", dash="dash"),  # Black dashed line
                name="Max Capacity",
            )
        )

        # Add an annotation for the max capacity line
        fig.add_annotation(
            x=self.df["step"].max(),  # Position the annotation at the end of the x-axis
            y=capacity_max,  # Align with the maximum capacity value
            text="Max Capacity",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="left",
            yanchor="bottom",
        )

        # Customize the layout
        fig.update_layout(
            title="Evolution of CO2 Storage in Factory",
        )

        # Customize layout
        fig.update_layout(
            template="ggplot2",
            yaxis_title=f"{title} (Tons of CO2)",
            showlegend=False,  # Remove legend
        )

        if return_html:
            # Convert the figure to an HTML string and save it to a variable
            plot_html = fig.to_html(
                full_html=False,
                div_id=f"plot-container-{self.generated_excel_files_index}",
            )
            return plot_html, None
        else:
            return fig

    def plot_factory_capacity_evolution_violin(self, return_html=True):
        title = f"{self.factory_column} Capacity"

        # Extract storage capacity
        self.df["storage_capacity"] = self.df[self.factory_column].apply(
            lambda x: x["capacity"] if x else None
        )

        # Calculate the median storage capacity
        median_capacity = self.df["storage_capacity"].median()

        # Extract capacity_max for percentage calculation and plotting
        capacity_max = (
            self.df[self.factory_column].iloc[0].get("capacity_max", 1)
        )  # Avoid division by zero
        percentage = (median_capacity / capacity_max) * 100 if capacity_max > 0 else 0

        # Prepare data for saving to Excel
        factory_capacity_data = pd.DataFrame(
            {"Step": self.df["step"], title: self.df["storage_capacity"]}
        )

        # Save the data to Excel
        self.save_sheets_in_memory([factory_capacity_data], title)
        self.graphs_data_dictionnary[f"{title} Evolution"] = (
            factory_capacity_data.to_dict(orient="records")
        )

        # Create a Plotly figure
        fig = go.Figure()

        # Add data to the plot

        fig = px.violin(
            self.df,
            y="storage_capacity",
            box=True,  # Show the box plot inside the violin
            points="outliers",  # Show outliers
            title=title,
        )

        if return_html:
            # Convert the figure to an HTML string and save it to a variable
            plot_html = fig.to_html(
                full_html=False,
                div_id=f"plot-container-{self.generated_excel_files_index}",
            )
            return plot_html, None
        else:
            return fig

        # Pass plot_html to your template for rendering

    def plot_travel_duration_evolution(self, return_html=True):
        # Function to calculate the duration of travel for each ship based on the navigation states
        def calculate_navigation_durations(df, ship_column):
            durations = []
            current_start_step = None
            current_start_state = None
            current_start_destination = None
            navigating_states = ["NAVIGATING", "DOCKING"]

            # Iterate over the rows of the dataframe to calculate travel durations
            for i, row in df.iterrows():
                ship = row[ship_column]
                step = row["step"]

                if ship and "state" in ship:
                    state = ship["state"]
                    destination = ship["destination"]

                    # Start of navigation phase: 'NAVIGATING' state and a valid destination
                    if (
                        state == "NAVIGATING"
                        and destination
                        and (current_start_state != "NAVIGATING")
                    ):
                        current_start_step = step
                        current_start_state = state
                        current_start_destination = destination

                    # End of navigation phase: 'DOCKED' state and matching destination
                    elif (
                        state == "DOCKED"
                        and destination == current_start_destination
                        and current_start_state == "NAVIGATING"
                    ):
                        duration = step - current_start_step
                        if duration > 0:  # Ensure valid duration
                            durations.append(duration)
                        current_start_state = None  # Reset for next phase

            return durations

        # Prepare lists to store the data for Excel
        self.travel_data = []
        all_ship_durations = {}

        # Loop through each ship column, calculate its travel durations and store for plotting
        for ship_column in self.ship_columns:
            ship_durations = calculate_navigation_durations(self.df, ship_column)
            all_ship_durations[ship_column] = ship_durations

            # Add the data for this ship to the travel_data list
            for i, duration in enumerate(ship_durations):
                self.travel_data.append(
                    {
                        "Ship": ship_column,
                        "Trip": i + 1,  # Phase corresponds to the index + 1
                        "Duration (hours)": duration,
                    }
                )

        # Create a Plotly figure for the travel durations over phases
        fig1 = go.Figure()

        # Add a line for each ship's travel durations
        for ship_column, durations in all_ship_durations.items():
            fig1.add_trace(
                go.Scatter(
                    x=list(
                        range(1, len(durations) + 1)
                    ),  # Phases are numbered starting from 1
                    y=durations,
                    mode="lines+markers",  # Line with markers at each phase
                    name=ship_column,
                )
            )

        # Customize the layout of the plot
        fig1.update_layout(
            template="ggplot2",
            title="Evolution of Ships' Journey Times",
            xaxis_title="Trips",
            yaxis_title="Duration (hours)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            legend_title="Ships",
        )

        # Save the travel durations data to Excel
        if self.travel_data:
            travel_df = pd.DataFrame(self.travel_data)
            self.save_sheets_in_memory([travel_df], "Ships journey times")

            # Store the data in the graph dictionary for later use in templates
            self.graphs_data_dictionnary["Evolution of ships journey times"] = (
                travel_df.to_dict(orient="records")
            )
        # Save the plot as HTML
        plot_html_1 = fig1.to_html(
            full_html=False, div_id=f"plot-container-{self.generated_excel_files_index}"
        )

        # Save the table as HTML
        # Save the table as HTML

        # Store plot HTML in dictionary for later use in templates
        self.graphs_data_dictionnary["Evolution of ships journey times"] = plot_html_1

        if return_html:
            return plot_html_1, None
        else:
            return fig1

    def calculate_waiting_durations(self, df, ship_column):
        # Function to calculate waiting times for each ship based on the states
        waiting_durations = []
        current_wait_start = None
        current_state = None
        current_destination = None

        waiting_states = ["WAITING", "DOCKED"]

        # Iterate over the rows of the dataframe to calculate waiting times
        for i, row in df.iterrows():
            ship = row[ship_column]
            step = row["step"]

            if ship and "state" in ship:
                state = ship["state"]
                destination = ship["destination"]

                # Start of waiting phase: state is 'WAITING' and there is a valid destination
                if state in waiting_states and current_state not in waiting_states:
                    current_wait_start = step
                    current_state = state
                    current_destination = destination

                # End of waiting phase: state changes from 'WAITING' to something else
                elif state not in waiting_states and current_state in waiting_states:
                    duration = step - current_wait_start
                    if duration > 0:  # Ensure valid duration
                        waiting_durations.append(duration)
                    current_state = None  # Reset for next phase

        return waiting_durations

    def plot_waiting_time_evolution(self, return_html=True):
        # Prepare lists to store the data for Excel
        self.waiting_data = []
        all_ship_waiting_times = {}
        # Loop through each ship column, calculate its waiting times and store for plotting
        for ship_column in self.ship_columns:
            waiting_times = self.calculate_waiting_durations(self.df, ship_column)
            all_ship_waiting_times[ship_column] = waiting_times

            # Add the data for this ship to the self.waiting_data list
            for i, waiting_time in enumerate(waiting_times):
                self.waiting_data.append(
                    {
                        "Ship": ship_column,
                        "Trip": i + 1,  # Phase corresponds to the index + 1
                        "Waiting Time (hours)": waiting_time,
                    }
                )

        # Create a Plotly figure for the waiting times over phases
        fig1 = go.Figure()

        # Add a line for each ship's waiting times
        for ship_column, waiting_times in all_ship_waiting_times.items():
            fig1.add_trace(
                go.Scatter(
                    x=list(
                        range(1, len(waiting_times) + 1)
                    ),  # Trips are numbered starting from 1
                    y=waiting_times,
                    mode="lines+markers",  # Line with markers at each phase
                    name=ship_column,
                )
            )

        # Customize the layout of the plot
        fig1.update_layout(
            template="ggplot2",
            title="Evolution of Ships' Waiting Times",
            xaxis_title="Trips",
            yaxis_title="Waiting Time (hours)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            legend_title="Ships",
        )

        # Save the waiting times data to Excel
        if self.waiting_data:
            waiting_df = pd.DataFrame(self.waiting_data)
            self.save_sheets_in_memory([waiting_df], "Ships Waiting Times")

            # Store the data in the graph dictionary for later use in templates
            self.graphs_data_dictionnary["Evolution of Ships Waiting Times"] = (
                waiting_df.to_dict(orient="records")
            )

        # Save the plot as HTML
        plot_html_1 = fig1.to_html(
            full_html=False, div_id=f"plot-container-{self.generated_excel_files_index}"
        )

        # Store plot HTML in dictionary for later use in templates
        self.graphs_data_dictionnary["Evolution of Ships Waiting Times"] = plot_html_1

        if return_html:
            return plot_html_1, None
        else:
            return fig1

    def calculate_total_co2(self):
        # Function to calculate total CO2 transported (proportional to the ship's capacity)
        total_co2 = []  # List to store the total CO2 (capacity) at each time step
        steps = []  # List to store the corresponding time steps

        # Iterate over the rows of the dataframe to sum the capacities at each time step
        for i, row in self.df.iterrows():
            total_capacity = 0
            # Sum the capacities of all ships (e.g., ship1, ship2, etc.)
            for ship_column in self.ship_columns:
                for s in self.config["ships"]:
                    if s["name"] == ship_column:
                        ship_max_capacity = s["capacity_max"]
                        break
                else:
                    ship_max_capacity = 0
                if ship_column in row and isinstance(row[ship_column], dict):
                    ship_data = row[ship_column]
                    total_capacity += float(ship_data.get("capacity", 0)) / float(
                        ship_max_capacity
                    )
            total_co2.append(total_capacity * 100)
            steps.append(row["step"])

        return steps, total_co2

    def plot_co2_transportation(self, return_html=True):
        # Calculate total CO2 transported over time
        steps, total_co2 = self.calculate_total_co2()

        # Create a Plotly figure for the total CO2 transported over time
        fig1 = go.Figure()

        # Add a line for total CO2 transported
        fig1.add_trace(
            go.Scatter(
                x=steps,
                y=total_co2,
                mode="lines",  # Line chart for CO2 evolution over time
                name="Total CO2 Transported",
                line=dict(color="blue"),
            )
        )

        # Customize the layout of the plot
        fig1.update_layout(
            template="ggplot2",
            title="Total CO2 Transported by Ships Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Total CO2 Transported (Tons of CO2)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
        )

        # Prepare data for saving to Excel
        co2_data = pd.DataFrame(
            {"Time Step": steps, "Total CO2 Transported": total_co2}
        )

        # Save the data to Excel
        self.save_sheets_in_memory([co2_data], "CO2 Transported Over Time")

        # Store the data in the graph dictionary for later use in templates
        self.graphs_data_dictionnary["Total CO2 Transported by Ships Over Time"] = (
            co2_data.to_dict(orient="records")
        )

        # Save the plot as HTML
        plot_html_1 = fig1.to_html(
            full_html=False, div_id=f"plot-container-{self.generated_excel_files_index}"
        )

        # Store plot HTML in dictionary for later use in templates
        self.graphs_data_dictionnary["Total CO2 Transported by Ships Over Time"] = (
            plot_html_1
        )

        if return_html:
            return plot_html_1, None
        else:
            return fig1

    def plot_storage_capacity_comparison(self, return_html=True):
        # Initialize variables to track the total hours for each condition
        total_hours_capacity_max_equals_capacity = 0
        total_hours_capacity_max_greater_than_capacity = 0
        # Initialize lists to store the data for saving to Excel
        data_for_excel = []

        # Loop through each row of the dataframe to evaluate the conditions
        for i, row in self.df.iterrows():
            # Get storage object for the row
            storage = row[self.factory_column]
            step = row["step"]

            # Ensure the storage object has the necessary attributes
            if storage and "capacity" in storage and "capacity_max" in storage:
                capacity = storage["capacity"]
                capacity_max = storage["capacity_max"]

                # Condition 1: capacity_max == capacity
                if capacity_max == capacity:
                    total_hours_capacity_max_equals_capacity += 1
                    data_for_excel.append(
                        {
                            "Step": step,
                            "Condition": "capacity_max == capacity",
                            "Capacity": capacity,
                            "Capacity Max": capacity_max,
                        }
                    )

                # Condition 2: capacity_max > capacity
                elif capacity_max > capacity:
                    total_hours_capacity_max_greater_than_capacity += 1
                    data_for_excel.append(
                        {
                            "Step": step,
                            "Condition": "capacity_max > capacity",
                            "Capacity": capacity,
                            "Capacity Max": capacity_max,
                        }
                    )

        # Prepare data for the bar chart
        categories = ["capacity_max == capacity", "capacity_max > capacity"]
        values = [
            total_hours_capacity_max_equals_capacity,
            total_hours_capacity_max_greater_than_capacity,
        ]

        # Create a Plotly bar chart
        fig1 = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker=dict(color=["blue", "orange"]),
                    text=values,  # Adds the values as labels on top of the bars
                    textposition="auto",
                )
            ]
        )

        columns = ["Step", "Condition", "Capacity", "Capacity Max"]
        columns = ["Step", "Condition", "Capacity", "Capacity Max"]
        values = {column: [row[column] for row in data_for_excel] for column in columns}

        # Add labels and title
        fig1.update_layout(
            template="ggplot2",
            title="Total Hours by Storage Capacity Conditions in Factory",
            xaxis_title="Condition",
            yaxis_title="Total Hours",
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            xaxis=dict(showgrid=False),
        )

        # Save the data for Excel
        if data_for_excel:
            storage_capacity_data = pd.DataFrame(data_for_excel)
            self.save_sheets_in_memory(
                [storage_capacity_data],
                "Storage Capacity Conditions",
            )
            self.graphs_data_dictionnary[
                "Total Hours by Storage Capacity Conditions in Factory"
            ] = storage_capacity_data.to_dict(orient="records")

        # Save the plot as HTML
        plot_html_1 = fig1.to_html(
            full_html=False, div_id=f"plot-container-{self.generated_excel_files_index}"
        )

        # Store plot HTML in dictionary for later use in templates
        self.graphs_data_dictionnary[
            "Total Hours by Storage Capacity Conditions in Factory"
        ] = plot_html_1

        if return_html:
            return plot_html_1, None
        else:
            return fig1

    def plot_factory_wasted_production_over_time(self, return_html=True):
        # Extract the 'wasted_production' values and the corresponding 'step' values
        wasted_production = []
        time_steps = []

        for i, row in self.df.iterrows():
            # Extract wasted production and time step for the factory
            factory_data = row[self.factory_column]
            step = row["step"]

            # Check if 'wasted_production' is available in the factory data
            if "wasted_production" in factory_data:
                wasted_production.append(factory_data["wasted_production"])
                time_steps.append(step)

        # Create a Plotly line plot (scatter plot with lines)
        fig1 = go.Figure(
            data=[
                go.Scatter(
                    x=time_steps,
                    y=wasted_production,
                    mode="lines",
                    line=dict(color="blue"),
                    name="Wasted Production (m³ of CO2)",
                )
            ]
        )

        # Add labels and title
        fig1.update_layout(
            template="ggplot2",
            title="Evolution of Wasted Production Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Wasted Production (m³ of CO2)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
        )

        # Prepare data for saving to Excel
        wasted_data = pd.DataFrame(
            {"Time Step": time_steps, "Wasted Production": wasted_production}
        )

        # Save the data to Excel
        self.save_sheets_in_memory([wasted_data], "Wasted CO2 production")

        # Store the data in the graph dictionary for later use in templates
        self.graphs_data_dictionnary["Evolution of Wasted CO2 production Over Time"] = (
            wasted_data.to_dict(orient="records")
        )

        # Save the plot as HTML
        plot_html_1 = fig1.to_html(
            full_html=False, div_id=f"plot-container-{self.generated_excel_files_index}"
        )

        # Store plot HTML in dictionary for later use in templates
        self.graphs_data_dictionnary["Evolution of Wasted CO2 production Over Time"] = (
            plot_html_1
        )

        if return_html:
            return plot_html_1, None
        else:
            return fig1

    def calculate_functional_kpis(self):
        # KPIs for the first table (Investissement de Départ)
        num_factory_tanks = self.df.iloc[0][self.factory_column].get("number_of_tanks")
        cost_per_tank = self.df.iloc[0][self.factory_column].get("cost_per_tank")
        scale_ratio = 1.2  # Example value

        tank_total_cost_in_factory = num_factory_tanks * cost_per_tank * scale_ratio

        # Calculate total ship costs dynamically
        total_ships_buying_costs = 0

        for ship_column in self.ship_columns:
            ship_data = self.df.iloc[0][ship_column]
            total_ships_buying_costs += float(ship_data.get("ship_buying_cost", 0))

        fuel_cost = 0
        ships_navigation_cost = 0
        ships_stoppage_cost = 0
        num_period_per_hours = self.config.get("general").get("num_period_per_hours")

        for i, row in self.df.iterrows():
            for ship_column in self.ship_columns:
                ship_navigation_cost = 0
                ship_stoppage_cost = 0
                if ship_column in row:
                    ship_data = row[ship_column]
                    state = ship_data.get("state", "")
                    fuel_consumption_per_day = float(
                        ship_data.get("fuel_consumption_per_day", 0)
                    )
                    if state in ["NAVIGATING", "DOCKING"]:
                        fuel_cost += (
                            fuel_consumption_per_day
                            * self.kpis["fuel_price_per_ton"]
                            / (num_period_per_hours * 24)
                        )  # Convert daily consumption to hourly
                        ship_navigation_cost += (
                            float(
                                ship_data.get("staff_cost_per_hour", 0)
                                * num_period_per_hours
                            )
                            + float(ship_data.get("usage_cost_per_hour", 0))
                            * num_period_per_hours
                        )
                        ship_stoppage_cost += (
                            float(
                                ship_data.get("staff_cost_per_hour", 0)
                                * num_period_per_hours
                            )
                            + float(ship_data.get("immobilization_cost_per_hour", 0))
                            * num_period_per_hours
                        )
                ships_navigation_cost += ship_navigation_cost
                ships_stoppage_cost += ship_stoppage_cost

        initial_investment = {
            "Storage Tank Purchase Cost": tank_total_cost_in_factory,
            "Boat Purchase Cost": total_ships_buying_costs,
        }

        # KPIs for the second table (Coûts Fonctionnels)
        ships_operating_cost = fuel_cost + ships_navigation_cost + ships_stoppage_cost

        last_state = self.df.iloc[-1]
        co2_capacity_stored = sum(
            [
                last_state[storage_column].get("received_co2_over_time", 0)
                for storage_column in self.storage_columns
            ]
        )

        co2_storage_cost = sum(
            [
                last_state[storage_column].get("storage_cost_per_m3", 0)
                * co2_capacity_stored
                for storage_column in self.storage_columns
            ]
        )

        co2_released_in_factory = sum(
            [i.get("wasted_production", 0) for i in self.df[self.factory_column]]
        )

        co2_release_cost = (
            co2_released_in_factory * self.kpis["co2_release_cost_per_ton"]
        )

        delay_penalty = 0.0  # Example delay penalty per step
        total_cost = ships_operating_cost + co2_release_cost + delay_penalty

        functional_costs = {
            "Fuel Cost": fuel_cost,
            "Boat Operational Costs": ships_operating_cost,
            "CO2 Storage Cost": co2_storage_cost,
            "co2_released_cost": co2_release_cost,
            "Boat Stoppage Cost": ships_stoppage_cost,
            "Navigation Cost": ships_navigation_cost,
            "Delay Cost": delay_penalty,
            "Total Cost": total_cost,
        }

        combined_total_cost = (
            initial_investment.get("Storage Tank Purchase Cost", 0)
            + initial_investment.get("Boat Purchase Cost", 0)
            + functional_costs.get("Total Cost", 0)
        )
        combined_cost = {"Combined Total Cost": combined_total_cost}

        self.combined_cost = combined_cost
        self.initial_investment = initial_investment
        self.functional_costs = functional_costs

        return initial_investment, functional_costs, combined_cost

    def plot_cost_kpis_table(self, return_html=True):
        # Data for the first table (Investissement de Départ)
        initial_investment, functional_costs, combined_cost = (
            self.calculate_functional_kpis()
        )

        # Data for the first table (Initial Investment)
        additional_data = [
            [
                "Storage Tank Purchase Cost",
                "Number of tanks × Cost per tank × Scaling ratio",
                self._format_costs(initial_investment["Storage Tank Purchase Cost"]),
            ],
            [
                "Boat Purchase Cost",
                "∑ Cost of boats",
                self._format_costs(initial_investment["Boat Purchase Cost"]),
            ],
        ]

        # Data for the second table (Operational Costs)
        data_with_values = [
            [
                "Fuel Cost",
                "Fuel consumed × Fuel price",
                self._format_costs(functional_costs["Fuel Cost"]),
            ],
            [
                "Boat Operational Costs",
                "Navigation cost + Stoppage cost + Fuel cost",
                self._format_costs(functional_costs["Boat Operational Costs"]),
            ],
            [
                "CO2 Storage Cost",
                "Quantity of CO2 stored × Storage cost per m³",
                self._format_costs(functional_costs["CO2 Storage Cost"]),
            ],
            [
                "Released CO2 Cost",
                "Quantity of CO2 released × Release cost per ton of CO2",
                self._format_costs(functional_costs["co2_released_cost"]),
            ],
            [
                "Delay Cost",
                "Delay time × Penalty for delays (NOT IMPLEMENTED)",
                self._format_costs(functional_costs["Delay Cost"]),
            ],
            [
                "Total Cost w\o Storage Costs",
                "∑ Previous costs",
                self._format_costs(functional_costs["Total Cost"]),
            ],
        ]

        # Prepare DataFrames for both tables
        additional_df = pd.DataFrame(
            additional_data, columns=["INITIAL INVESTMENT", "FORMULA", "VALUE"]
        )
        functional_df = pd.DataFrame(
            data_with_values, columns=["FUNCTIONAL COST", "FORMULA", "VALUE"]
        )

        combined_cost_df = pd.DataFrame(
            [
                [
                    "Combined Total Cost w\o Storage costs",
                    "Investement Costs + Functional costs",
                    self._format_costs(combined_cost["Combined Total Cost"]),
                ]
            ],
            ["COMBINED COST", "FORMULA", "VALUE"],
        )

        # Create Plotly figure for the tables
        fig = make_subplots(
            rows=3,  # 3 rows
            cols=1,  # 1 column
            shared_xaxes=False,  # Each table will have its own set of cells
            vertical_spacing=0.02,  # Space between rows
            row_heights=[0.2, 0.66, 0.2],  # Adjust the relative height of the rows
            specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "table"}]],
        )

        # Table 1: Initial Investment
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["INITIAL INVESTMENT", "FORMULA", "VALUE"],
                    fill_color="paleturquoise",
                    align="center",
                ),
                cells=dict(
                    values=[additional_df[col] for col in additional_df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            ),
            row=1,
            col=1,
        )

        # Table 2: Functional Costs
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["FUNCTIONAL COST", "FORMULA", "VALUE"],
                    fill_color="paleturquoise",
                    align="center",
                ),
                cells=dict(
                    values=[functional_df[col] for col in functional_df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            ),
            row=2,
            col=1,
        )

        # Table 3: Combined Costs
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["COMBINED COST", "FORMULA", "VALUE"],
                    fill_color="paleturquoise",
                    align="center",
                ),
                cells=dict(
                    values=combined_cost_df.iloc[0],
                    fill_color="lavender",
                    align="left",
                ),
            ),
            row=3,
            col=1,
        )

        # Update the layout
        fig.update_layout(
            template="ggplot2",
            title="KPIs Table: Financial Overview",
            height=800,
            showlegend=False,
        )

        # Save the Plotly figure as an HTML file
        kpis_html = fig.to_html(full_html=False, div_id="KPIS_fina_table")

        # Store the plot HTML for later use
        self.graphs_data_dictionnary["KPIs_Financières"] = kpis_html

        self.save_sheets_in_memory([additional_df], "Initial Investment")
        self.save_sheets_in_memory([functional_df], "Operational Costs")

        # Store the data in the graph dictionary for later use in templates
        self.graphs_data_dictionnary["KPIs_Financières"] = [
            additional_df.to_dict(orient="records"),
            functional_df.to_dict(orient="records"),
        ]

        if return_html:
            return kpis_html, (initial_investment, functional_costs, combined_cost)
        else:
            return fig

    def plot_metric_kpis_table(self, return_html=True):
        co2_vented_quantity = sum(
            [i.get("wasted_production", 0) for i in self.df[self.factory_column]]
        )
        co2_transported_quantity = sum(
            [
                self.df.iloc[-1][storage_column].get("received_co2_over_time", 0)
                for storage_column in self.storage_columns
            ]
        )
        average_travel_duration = np.mean(
            [trip.get("Duration (hours)", 0) for trip in self.travel_data]
        )
        average_waiting_time = np.mean(
            [trip.get("Waiting Time (hours)", 0) for trip in self.waiting_data]
        )
        time_tank_full = np.sum(
            [
                1 if i.get("capacity", 0) >= i.get("capacity_max", 0) else 0
                for i in self.df[self.factory_column]
            ]
        )
        time_tank_not_full = np.sum(
            [
                1 if i.get("capacity", 0) < i.get("capacity_max", 0) else 0
                for i in self.df[self.factory_column]
            ]
        )
        percentage_time_tank_full = (
            time_tank_full / (time_tank_full + time_tank_not_full) * 100
        )
        percentage_co2_vented = (
            co2_vented_quantity / (co2_vented_quantity + co2_transported_quantity) * 100
        )

        num_rows = 1

        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=False,  # Each table will have its own set of cells
            vertical_spacing=0.02,  # Space between rows
            specs=[[{"type": "table"}] * num_rows],
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Label", "Value"],
                    fill_color="paleturquoise",
                    align="center",
                ),
                cells=dict(
                    values=pd.DataFrame(
                        [
                            [
                                "Ships average travel time",
                                "Ships average waiting time",
                                "CO2 vented quantities(tons)",
                                "CO2 stored quantities(tons)",
                                "Time storage are full (hours)",
                                "Time storage are not full (hours)",
                            ],
                            [
                                self._format_time(average_travel_duration),
                                self._format_time(average_waiting_time),
                                f"{self._format_quantity(co2_vented_quantity)} ({percentage_co2_vented:,.2f} %)",
                                f"{self._format_quantity(co2_transported_quantity)} ({100-percentage_co2_vented:,.2f} %)",
                                f"{self._format_time(time_tank_full)} ({percentage_time_tank_full:,.2f} %)",
                                f"{self._format_time(time_tank_not_full)} ({100-percentage_time_tank_full:,.2f} %)",
                            ],
                        ]
                    ),
                    fill_color="lavender",
                    align="right",
                ),
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            template="ggplot2",
            title="KPIs Table: Metrics Overview",
            height=800,
            showlegend=False,
        )

        # Save the Plotly figure as an HTML file
        kpis_html = fig.to_html(full_html=False, div_id="KPIS_metrics_table")

        if return_html:
            return kpis_html, None
        else:
            return fig

    def parse_dict(self, value):
        try:
            return ast.literal_eval(value)
        except:
            return None

    def generate_kpis_graphs(self):
        print(
            f"""{Fore.CYAN}Plotting KPIs Graphs and saving data to an excel file...{Style.RESET_ALL}"""
        )
        html_plots = []
        html_plots.extend(
            [
                self.plot_factory_capacity_evolution(),
                self.plot_factory_capacity_evolution_violin(),
                self.plot_storage_capacity_comparison(),
                self.plot_factory_wasted_production_over_time(),
                self.plot_travel_duration_evolution(),
                self.plot_waiting_time_evolution(),
                self.plot_co2_transportation(),
                self.plot_cost_kpis_table(),
                self.plot_metric_kpis_table(),
            ]
        )

        KpisGraphsGenerator.generate_html_page_with_plots(
            [i for i, _ in html_plots], self.output_folder
        )
        # self.save_all_sheets_to_excel()

        kpis = self.plot_cost_kpis_table()[1]
        return kpis

    @staticmethod
    def generate_html_page_with_plots(html_plots, output_folder, full_page=False):

        if not full_page:
            # Determine the path to the bundled directory
            if getattr(sys, "frozen", False):
                # If running as a bundled .exe
                templates_base_path = os.path.join(
                    sys._MEIPASS, "KPIs"
                )  # Temporary folder where PyInstaller stores the bundled files
            else:
                # If running from the source code
                templates_base_path = os.path.dirname(__file__)

            # Setup Jinja2 environment and load the template
            template_loader = FileSystemLoader(
                searchpath=os.path.join(templates_base_path)
            )
            template_env = Environment(loader=template_loader)
            template = template_env.get_template("template.html")

            # Render the HTML with the images
            html_content = template.render({"plots": html_plots})
        else:
            html_content = "".join(html_plots)

        # Write HTML content to a file

        os.makedirs(output_folder, exist_ok=True)
        with open(
            os.path.join(output_folder, "Graphs gallery.html"), "w", encoding="utf-8"
        ) as html_file:
            html_file.write(html_content)

        print(f"HTML file generated: {output_folder}")

    @staticmethod
    def plot_factories_capacity_evolution(simulations):
        """
        Plot separate violin charts for multiple ports, combining storage capacities into a single chart for each port.

        :param dataframes: Dictionary where keys are ports and values are lists of DataFrames for each scenario.
        :param labels: Dictionary where keys are ports and values are lists of labels corresponding to each DataFrame.
        """

        simulations_by_ports = {
            "Bergen": [i for i in simulations if "bergen" in i.config_name],
            "Rotterdam": [i for i in simulations if "rotterdam" in i.config_name],
            "other": [
                i
                for i in simulations
                if ("bergen" not in i.config_name)
                and ("rotterdam" not in i.config_name)
            ],
        }

        combined_df = pd.DataFrame()

        for label, port_simulations in simulations_by_ports.items():
            # Get corresponding labels
            for simulation in port_simulations:
                # Extract capacity
                df = simulation.data_df
                label = simulation.config_name
                factory_column = simulation.config["factory"]["name"]
                df["storage_capacity"] = df[factory_column].apply(
                    lambda x: x["capacity"] if x else None
                )
                # Add the port and scenario label for separation
                df["Scenario"] = label
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            if combined_df.empty:
                continue
        # Create a violin chart using Plotly Express
        fig = px.violin(
            combined_df,
            x="storage_capacity",
            color="Scenario",  # Differentiate by scenario
            box=True,  # Show the box plot inside the violin
            points="outliers",  # Show outliers
            title=f"Distribution of CO2 Storage in Le Havre accross different scenarios",
        )

        # Customize layout
        fig.update_layout(
            template="ggplot2",
            xaxis_title="Storage Capacity (Tons of CO2)",
            xaxis_range=[-1000, 35000],
            annotations=[
                dict(
                    x=0.5,
                    y=-0.2,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    text=(
                        "Hover over the chart to view specific details about storage capacity. "
                        "The legend differentiates scenarios for easy comparison."
                    ),
                    font=dict(size=12),
                    align="center",
                )
            ],
            legend={"traceorder": "reversed"},
        )
        # Add a descriptive legend title
        fig.update_layout(
            legend_title=dict(
                text="Double click a Scenario in the Legend to highlight relevant Data",
                font=dict(size=12, color="black"),
            )
        )
        fig.update_traces(box_visible=True, meanline_visible=True)

        # Save the figure in the dictionary
        return fig.to_html()

    @staticmethod
    def plot_summary_table_for_each_scenario(simulations):
        df_columns = [
            "Scenario",
            "Number of Boats",
            "Boat Purchase Cost",
            "CO2 Released Cost",
            "Total Cost",
        ]
        combined_df = pd.DataFrame(columns=df_columns)
        for simulation in simulations:
            df = simulation.data_df
            kpi = simulation.kpis
            config = simulation.config_name
            nb_boats = len([col for col in df.columns if "ship" in col.lower()])
            boat_cost = kpi[0]["Boat Purchase Cost"]
            co2_released_cost = kpi[1]["co2_released_cost"]
            total_cost = kpi[2]["Combined Total Cost"]

            serie = pd.Series(
                {
                    "Scenario": config,
                    "Number of Boats": nb_boats,
                    "Boat Purchase Cost": KpisGraphsGenerator._format_costs(boat_cost),
                    "CO2 Released Cost": KpisGraphsGenerator._format_costs(
                        co2_released_cost
                    ),
                    "Total Cost": KpisGraphsGenerator._format_costs(total_cost),
                }
            )

            combined_df = pd.concat(
                [combined_df, serie.to_frame().T], ignore_index=True
            )
        num_rows = 1
        fig = make_subplots(
            rows=num_rows,  # 1 rows
            cols=1,  # 1 column
            shared_xaxes=False,  # Each table will have its own set of cells
            specs=[[{"type": "table"}] * num_rows],
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=df_columns,
                    fill_color="paleturquoise",
                    align="center",
                ),
                cells=dict(
                    values=[combined_df[col] for col in combined_df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            ),
            row=1,
            col=1,
        )

        return fig.to_html(full_html=False, div_id="Summary KPIs")

    @staticmethod
    def plot_summary_costs_for_each_scenario(simulations):
        df_columns = [
            "Scenario",
            "Type of cost",
            "Cost",
        ]
        costs_to_display = [
            "Storage Tank Purchase Cost",
            "Boat Purchase Cost",
            "Fuel Cost",
            "Boat Operational Costs",
            "co2_released_cost",
            "Delay Cost",
        ]
        combined_df = pd.DataFrame(columns=df_columns)
        for simulation in simulations:
            costs = (
                simulation.generator.initial_investment
                | simulation.generator.functional_costs
                | simulation.generator.combined_cost
            )
            for cost in costs_to_display:
                serie = pd.Series(
                    {
                        df_columns[0]: simulation.config_name,
                        df_columns[1]: cost,
                        df_columns[2]: costs[cost],
                    }
                )

                combined_df = pd.concat(
                    [combined_df, serie.to_frame().T], ignore_index=True
                )

        fig = px.bar(
            combined_df,
            x=df_columns[0],
            y=df_columns[2],
            color=df_columns[1],
            title="Costs Breakdown",
        )

        return fig.to_html(full_html=False, div_id="Costs breakdown")
