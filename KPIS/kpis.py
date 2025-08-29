import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from eco2_normandy.logger import Logger
from KPIS.utils import to_MultiIndex_dfs

pio.templates.default = "ggplot2"


class Kpis:
    def __init__(self, simulation_df, config, logger=None):
        self.config = config
        self.logger = logger or Logger()
        self.factory_name = self.config["factory"]["name"]
        self.storage_names = [s["name"] for s in self.config["storages"]]
        self.ship_names = [s["name"] for s in self.config["ships"]]
        self.figsize = (9, 4)
        self.kpis = config["KPIS"]
        self.num_period = self.config["general"]["num_period"]
        self.dfs = simulation_df
        self.config["general"]["number_of_ships"] = len(self.config["ships"])
        self.trips = self._trip_analysis()

    def get_non_numpy_sol(self, dic):
        return {k: float(v) for k, v in dic.items()}

    # TODO: look like not needed
    def safe_float_conversion(self, val, default=0.0):
        """
        Safely convert a value to float, handling pandas Series from MultiIndex DataFrames.

        Args:
            val: Value to convert (could be scalar, Series, etc.)
            default: Default value if conversion fails

        Returns:
            float: Converted value
        """
        if isinstance(val, pd.Series):
            return float(val.iloc[0] if len(val) > 0 else default)
        else:
            return float(val)

    def _to_MultiIndex_dfs(self, dic):
        return to_MultiIndex_dfs(dic)

    def get_lvl_0_index(self, dfs):
        return self._get_lvl_index(dfs, 0)

    def get_lvl_1_index(self, dfs):
        return self._get_lvl_index(dfs, 1)

    def _get_lvl_index(self, dfs, n):
        return dfs.columns.get_level_values(n).unique()

    def _get_states_to_dfs(self, states: list):
        ships_states = {}
        for s in self.get_lvl_0_index(self.trips):
            df = self.trips[s].fillna(0)
            state_df = df[states]
            ships_states[s] = state_df
        return self._to_MultiIndex_dfs(ships_states)

    def _trip_analysis(self) -> pd.DataFrame:
        """Analyze trips data for each ship. 1 trip is consider each time a ship leaves factory.

        Returns:
            pd.DataFrame: DataFrame containing trip analysis results.
        """
        ships = [s["name"] for s in self.config["ships"]]

        all_trips = {}
        states = ["DOCKED", "DOCKING", "WAITING", "NAVIGATING", "LOADING", "UNLOADING"]
        init_trips = lambda: {state: 0 for state in states}
        for ship_name in ships:
            df_ship = self.dfs[ship_name]

            trips_list = []
            current_trip = init_trips()

            for _, row in df_ship.iterrows():
                state = str(row["state"])
                destination_name = row["destination"]
                capacity = row.get("capacity")

                if state == "DOCKED":
                    # Condition de fin d'un trajet
                    n_trips = sum(current_trip[s] for s in states)
                    if destination_name == self.factory_name and capacity == 0 and n_trips > 2:
                        trips_list.append(current_trip)
                        # Reset des compteurs pour un nouveau trajet
                        current_trip = init_trips()

                    if destination_name == self.factory_name:
                        current_trip["LOADING"] += 1
                    elif destination_name in self.storage_names:
                        current_trip["UNLOADING"] += 1

                current_trip[state] += 1
            all_trips[ship_name] = trips_list
        self.trips = self._to_MultiIndex_dfs(all_trips)
        self.trips.index = pd.Index([f"Trip {i + 1}" for i in range(len(self.trips))])
        return self.trips

    def factory_filling_rate(self):
        df = self.dfs[self.factory_name]
        capa_mean = df.capacity.mean()
        capa_max = df["capacity_max"].iloc[0]
        return capa_mean / capa_max

    def storage_filling_rate(self):
        storage_rates = {}
        for storage_name in self.storage_names:
            df = self.dfs[storage_name]
            capa_mean = df.capacity.mean()
            capa_max = df["capacity_max"].iloc[0]
            storage_rates[storage_name] = capa_mean / capa_max
        return pd.Series(storage_rates).mean()

    def wasted_production(self):
        return self.dfs[self.factory_name]["wasted_production"].sum()

    def wasted_production_over_time(self):
        return self.dfs[self.factory_name]["wasted_production"].cumsum()

    def get_waiting_time_dfs(self, waiting_states=["DOCKED", "WAITING"]):
        return self._get_states_to_dfs(waiting_states)

    def get_total_waiting_time(self):
        dfs = self.get_waiting_time_dfs()
        return dfs.sum().sum()

    def get_navigating_time_dfs(self, navigation_states=["NAVIGATING"]):
        return self._get_states_to_dfs(navigation_states)

    def get_total_navigating_time(self):
        dfs = self.get_navigating_time_dfs()
        return dfs.sum().sum()

    def calculate_functional_kpis(self):
        """
        Calculate KPIs based on the simulation dataframe.
        Returns a dictionary with the calculated KPIs.
        """
        kpis = {}
        factory_df = self.dfs[self.factory_name]
        storages_df = self.dfs[self.storage_names]
        ships_df = self.dfs[self.ship_names]

        initial_row_factory = factory_df.iloc[0]
        num_factory_tanks = self.safe_float_conversion(initial_row_factory["number_of_tanks"])
        cost_per_tank = self.safe_float_conversion(initial_row_factory["cost_per_tank"])
        scale_ratio = 1.2
        tank_total_cost_in_factory = num_factory_tanks * cost_per_tank * scale_ratio

        initial_row_ship = ships_df[self.ship_names[0]].iloc[0]
        total_ships_buying_costs = (
            self.safe_float_conversion(initial_row_ship["ship_buying_cost"]) * self.config["general"]["number_of_ships"]
        )

        initial_investment = {
            "Storage Tank Purchase Cost": tank_total_cost_in_factory,
            "Boat Purchase Cost": total_ships_buying_costs,
        }

        # Operational Costs: use trips MultiIndex DataFrame to aggregate activity counts
        num_period_per_hours = self.config["general"]["num_period_per_hours"]
        fuel_cost = 0
        ships_navigation_cost = 0
        ships_stoppage_cost = 0

        # Sum all trips counts across steps (each cell is a count)
        trips_sum = self.trips.sum()

        for ship in self.ship_names:
            n_nav = trips_sum.get((ship, "NAVIGATING"), 0)
            n_dock = trips_sum.get((ship, "DOCKING"), 0)
            n_waiting = trips_sum.get((ship, "WAITING"), 0)
            n_actions = n_nav + n_dock
            days_navigating = n_actions / (num_period_per_hours * 24)

            # Get cost parameters from the first row of the corresponding ships dataframe
            ship_params = ships_df[ship].iloc[0]
            fuel_consumption = self.safe_float_conversion(ship_params.get("fuel_consumption_per_day", 0))
            staff_cost = self.safe_float_conversion(ship_params.get("staff_cost_per_hour", 0))
            usage_cost = self.safe_float_conversion(ship_params.get("usage_cost_per_hour", 0))
            immobilization_cost = self.safe_float_conversion(ship_params.get("immobilization_cost_per_hour", 0))

            fuel_cost += days_navigating * fuel_consumption * self.kpis.get("fuel_price_per_ton", 0)
            ships_navigation_cost += n_actions / num_period_per_hours * (staff_cost + usage_cost)
            ships_stoppage_cost += n_waiting / num_period_per_hours * (staff_cost + immobilization_cost)
        ships_operating_cost = fuel_cost + ships_navigation_cost + ships_stoppage_cost

        # Storage Cost: accumulate CO2 received and related storage costs from last row of each storage's dataframe
        co2_capacity_stored = 0
        co2_storage_cost = 0
        for storage in self.storage_names:
            last_storage_state = storages_df[storage].iloc[-1]
            received_co2 = self.safe_float_conversion(last_storage_state.get("received_co2_over_time", 0))
            storage_cost_rate = self.safe_float_conversion(last_storage_state.get("storage_cost_per_m3", 0))
            co2_capacity_stored += received_co2
            co2_storage_cost += received_co2 * storage_cost_rate

        # Factory CO2 Release Cost: use last row of the factory dataframe
        co2_released_in_factory = self.wasted_production()
        co2_release_cost = (
            co2_released_in_factory
            * self.config["general"].get("m3_to_tons", 0.9)
            * self.kpis.get("co2_release_cost_per_ton", 1)
        )

        # Assume delay penalty is zero for now
        delay_penalty = 0.0

        functional_costs = {
            "Fuel Cost": fuel_cost,
            "Boat Operational Costs": ships_operating_cost,
            "Boat Stoppage Cost": ships_stoppage_cost,
            "Navigation Cost": ships_navigation_cost,
            "CO2 Storage Cost": co2_storage_cost,
            "co2_released_cost": co2_release_cost,
            "Delay Cost": delay_penalty,
            "Total Cost": ships_operating_cost + co2_release_cost + delay_penalty + co2_storage_cost,
        }

        combined_total_cost = (
            initial_investment["Storage Tank Purchase Cost"]
            + initial_investment["Boat Purchase Cost"]
            + functional_costs["Total Cost"]
        )

        # Populate the KPIs dictionary
        kpis["Initial Investment"] = initial_investment
        kpis["Functional Costs"] = functional_costs
        kpis["Combined Total Cost"] = combined_total_cost

        return kpis

    def plot_factory_capacity_evolution(self):
        factory_df = self.dfs[self.factory_name]
        capacity_max = self.config["factory"]["capacity_max"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=factory_df.index,
                y=factory_df["capacity"],
                mode="lines+markers",
                name="Factory Capacity",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[factory_df.index.min(), factory_df.index.max()],
                y=[capacity_max] * 2,
                mode="lines",
                name="Capacity max",
                line=dict(color="black", dash="dash"),  # Black dashed line
            )
        )
        fig.add_annotation(
            x=factory_df.index.max(),  # Position the annotation at the end of the x-axis
            y=capacity_max,  # Align with the maximum capacity value
            text="Max Capacity",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="left",
            yanchor="bottom",
        )

        fig.update_layout(
            title="Evolution of CO2 Storage in Factory",
        )

        # Customize layout
        fig.update_layout(
            template="ggplot2",
            yaxis_title=f"Le Havre capacity (Tons of CO2)",
            showlegend=False,
        )

        return fig

    def plot_factory_capacity_evolution_violin(self):
        factory_df = self.dfs[self.factory_name]

        fig = px.violin(
            y=factory_df["capacity"],
            box=True,  # Show the box plot inside the violin
            points="outliers",  # Show outliers
            title="Distribution of CO2 Storage in Factory",
        )

        fig.update_layout(
            template="ggplot2",
            yaxis_title="Factory Capacity (Tons of CO2)",
            xaxis_title="",
            showlegend=False,
        )

        return fig

    def plot_storage_capacity_comparison(self):
        factory_df = self.dfs[self.factory_name]

        capa_equals_max = (factory_df["capacity"] == factory_df["capacity_max"]).sum()
        capa_less_than_max = len(factory_df) - capa_equals_max

        categories = ["capacity == capacity_max", "capacity < capacity_max"]
        values = [capa_equals_max, capa_less_than_max]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker=dict(color=["blue", "orange"]),
                    text=values,
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            template="ggplot2",
            title="Total Hours by Storage Capacity Conditions in Factory",
            xaxis_title="Condition",
            yaxis_title="Total Hours",
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            xaxis=dict(showgrid=False),
        )

        return fig

    def plot_factory_wasted_production_over_time(self):
        wasted_cumsum = self.wasted_production_over_time()

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=wasted_cumsum.index,
                    y=wasted_cumsum,
                    mode="lines",
                    line=dict(color="blue"),
                    name="Cumulative Wasted Production (m³ of CO2)",
                )
            ]
        )

        fig.update_layout(
            template="ggplot2",
            title="Evolution of Cumulative Wasted Production Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Cumulative Wasted Production (m³ of CO2)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            showlegend=False,
        )

        return fig

    def plot_travel_duration_evolution(self):
        fig = go.Figure()

        # Extract navigation durations from the trips dataframe
        for ship_name in self.ship_names:
            if ship_name in self.get_lvl_0_index(self.trips):
                # Get navigation times per trip for this ship
                ship_nav_data = self.trips[ship_name]["NAVIGATING"]
                nav_durations = ship_nav_data[ship_nav_data > 0]

                if len(nav_durations) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(nav_durations) + 1)),
                            y=nav_durations,
                            mode="lines+markers",
                            name=ship_name,
                        )
                    )

        fig.update_layout(
            template="ggplot2",
            title="Evolution of Ships' Journey Times",
            xaxis_title="Trips",
            yaxis_title="Duration (hours)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            legend_title="Ships",
        )

        return fig

    def plot_waiting_time_evolution(self, waiting_states: list[str] = ["WAITING", "DOCKED"]):
        fig = go.Figure()

        # Extract waiting times from the trips dataframe
        for ship_name in self.ship_names:
            if ship_name in self.trips.columns.get_level_values(0):
                waiting_times = sum(self.trips[ship_name][state] for state in waiting_states)
                # Filter out zero values
                waiting_times = waiting_times[waiting_times > 0]

                if len(waiting_times) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(waiting_times) + 1)),
                            y=waiting_times,
                            mode="lines+markers",
                            name=ship_name,
                        )
                    )

        fig.update_layout(
            template="ggplot2",
            title="Evolution of Ships' Waiting Times",
            xaxis_title="Trips",
            yaxis_title="Waiting Time (hours)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            legend_title="Ships",
        )

        return fig

    def plot_co2_transportation(self, combine_ships: bool = False):
        ship_capacities = {ship["name"]: ship["capacity_max"] for ship in self.config["ships"]}

        # Calculate total CO2 as percentage across all ships
        fig = go.Figure()
        ships_transportation = []
        for ship_name in self.ship_names:
            total_capacity_pct = pd.Series(0.0, index=self.dfs[self.ship_names[0]].index)
            ship_df = self.dfs[ship_name]
            ship_max_capacity = ship_capacities[ship_name]
            total_capacity_pct = (ship_df["capacity"] / ship_max_capacity) * 100
            ships_transportation.append(total_capacity_pct)

        for i, ship_name in enumerate(self.ship_names):
            if combine_ships:
                transportation = sum(ships_transportation)
            else:
                transportation = ships_transportation[i]
            fig.add_trace(
                go.Scatter(
                    x=transportation.index,
                    y=transportation,
                    mode="lines+markers",
                    name=f"{ship_name} Total CO2 Transported",
                )
            )
            if combine_ships:
                break

        fig.update_layout(
            template="ggplot2",
            title="Total CO2 Transported by Ships Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Total CO2 Transported (% of ship capacity)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor="LightGrey"),
            showlegend=not combine_ships,
            legend_title="Ships",
        )

        return fig

    @staticmethod
    def _format_costs(val):
        return "{:,.2f} €".format(round(val, 2))

    @staticmethod
    def _format_time(val):
        return "{:d} H".format(round(val))

    @staticmethod
    def _format_quantity(val):
        return "{:d} Tons".format(round(val))

    def plot_cost_kpis_table(self):
        kpis = self.calculate_functional_kpis()
        initial_investment = kpis["Initial Investment"]
        functional_costs = kpis["Functional Costs"]
        combined_cost = kpis["Combined Total Cost"]

        # Data for Initial Investment table
        initial_investment_data = [
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

        # Data for Functional Costs table
        functional_costs_data = [
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
                "Total Cost w\\o Storage Costs",
                "∑ Previous costs",
                self._format_costs(functional_costs["Total Cost"]),
            ],
        ]

        # Combined cost data
        combined_cost_data = [
            [
                "Combined Total Cost w\\o Storage costs",
                "Investment Costs + Functional costs",
                self._format_costs(combined_cost),
            ]
        ]

        # Create DataFrames
        initial_investment_df = pd.DataFrame(
            initial_investment_data, columns=["INITIAL INVESTMENT", "FORMULA", "VALUE"]
        )
        functional_costs_df = pd.DataFrame(functional_costs_data, columns=["FUNCTIONAL COST", "FORMULA", "VALUE"])
        combined_cost_df = pd.DataFrame(combined_cost_data, columns=["COMBINED COST", "FORMULA", "VALUE"])

        # Create subplots for tables
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.02,
            row_heights=[0.2, 0.66, 0.2],
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
                    values=[initial_investment_df[col] for col in initial_investment_df.columns],
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
                    values=[functional_costs_df[col] for col in functional_costs_df.columns],
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
                    values=[combined_cost_df[col] for col in combined_cost_df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            ),
            row=3,
            col=1,
        )

        # Update layout
        fig.update_layout(
            template="ggplot2",
            title="KPIs Table: Financial Overview",
            height=800,
            showlegend=False,
        )

        return fig

    def plot_metric_kpis_table(self):
        factory_df = self.dfs[self.factory_name]

        # Calculate metrics
        co2_vented_quantity = self.wasted_production()

        # CO2 transported quantity (from last state of storages) - vectorized
        co2_transported_quantity = sum(
            self.dfs[storage_name]["received_co2_over_time"].iloc[-1] for storage_name in self.storage_names
        )

        # Calculate average travel and waiting times using the trip analysis - vectorized
        trips_sum = self.trips.sum()

        # Use pandas operations for faster calculations
        total_navigation_time = sum(trips_sum.get((ship, "NAVIGATING"), 0) for ship in self.ship_names)
        total_waiting_time = sum(
            trips_sum.get((ship, "WAITING"), 0) + trips_sum.get((ship, "DOCKED"), 0) for ship in self.ship_names
        )

        # Convert to hours (assuming periods are in hours)
        average_travel_duration = total_navigation_time / len(self.ship_names) if self.ship_names else 0
        average_waiting_time = total_waiting_time / len(self.ship_names) if self.ship_names else 0

        # Time tank full vs not full - use pandas vectorized operations
        time_tank_full = (factory_df["capacity"] >= factory_df["capacity_max"]).sum()
        time_tank_not_full = len(factory_df) - time_tank_full

        # Calculate percentages
        total_time = time_tank_full + time_tank_not_full
        percentage_time_tank_full = (time_tank_full / total_time * 100) if total_time > 0 else 0

        total_co2 = co2_vented_quantity + co2_transported_quantity
        percentage_co2_vented = (co2_vented_quantity / total_co2 * 100) if total_co2 > 0 else 0

        # Create metrics table
        metrics_data = [
            [
                "Ships average travel time",
                "Ships average waiting time",
                "CO2 vented quantities(tons)",
                "CO2 stored quantities(tons)",
                "Time storage are full (hours)",
                "Time storage are not full (hours)",
            ],
            [
                f"{average_travel_duration:.1f}",
                f"{average_waiting_time:.1f}",
                f"{self._format_quantity(co2_vented_quantity)} ({percentage_co2_vented:.2f} %)",
                f"{self._format_quantity(co2_transported_quantity)} ({100 - percentage_co2_vented:.2f} %)",
                f"{self._format_time(time_tank_full)} ({percentage_time_tank_full:.2f} %)",
                f"{self._format_time(time_tank_not_full)} ({100 - percentage_time_tank_full:.2f} %)",
            ],
        ]

        # Create figure
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "table"}]],
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Label", "Value"],
                    fill_color="paleturquoise",
                    align="center",
                ),
                cells=dict(
                    values=metrics_data,
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

        return fig
