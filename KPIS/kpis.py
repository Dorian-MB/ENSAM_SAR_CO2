import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd

from eco2_normandy.logger import Logger
from KPIS.utils import to_MultiIndex_dfs

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
        return {k:float(v) for k, v in dic.items()}
    
    # TODO: check if working
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

    def _trip_analysis(self):
        ships = [s["name"] for s in self.config["ships"]]

        all_trips = {}
        states = ["DOCKED", "DOCKING", "WAITING", "NAVIGATING", "LOADING", "UNLOADING"]
        init_trips = lambda: {state: 0 for state in states}
        for ship_name in ships:
            df_ship = self.dfs[ship_name]

            trips_list = []
            current_trip = init_trips()

            for _, row in df_ship.iterrows():
                state = str(row['state'])
                destination_name = row['destination']
                capacity = row.get('capacity')

                if state == "DOCKED" :
                    # Condition de fin d'un trajet
                    n_trips = sum(current_trip[s] for s in states)
                    if (destination_name == self.factory_name 
                        and capacity == 0
                        and n_trips > 2
                        ):
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
        self.trips.index = pd.Index([f"Trip {i+1}" for i in range(len(self.trips))])
        return self.trips

    def factory_filling_rate(self):
        df = self.dfs[self.factory_name]
        capa_mean = df.capacity.mean()
        capa_max = df["capacity_max"].iloc[0]
        return capa_mean/capa_max
    
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

    def _get_states_to_dfs(self, states:list): 
        ships_states = {}
        for s in self.get_lvl_0_index(self.trips):
            df = self.trips[s].fillna(0)
            state_df = df[states]
            ships_states[s] = state_df
        return self._to_MultiIndex_dfs(ships_states)

    def get_waiting_time_dfs(self):
        waiting_states = ["DOCKED", "WAITING"]
        return self._get_states_to_dfs(waiting_states)
    
    def get_total_waiting_time(self):
        dfs = self.get_waiting_time_dfs()
        return dfs.sum().sum()

    def get_navigating_time_dfs(self):
        navigation_state = ["NAVIGATING"]
        return self._get_states_to_dfs(navigation_state)

    def get_total_navigating_time(self):
        dfs = self.get_navigating_time_dfs()
        return dfs.sum().sum()

    # New versio of calculate_functional_kpis using MultiIndex DataFrame
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
        num_factory_tanks = self.safe_float_conversion(initial_row_factory.get("number_of_tanks", 1))
        cost_per_tank = self.safe_float_conversion(initial_row_factory.get("cost_per_tank", 1))
        scale_ratio = 1.2
        tank_total_cost_in_factory = num_factory_tanks * cost_per_tank * scale_ratio

        initial_row_ship = ships_df[self.ship_names[0]].iloc[0]
        total_ships_buying_costs = self.safe_float_conversion(initial_row_ship.get("ship_buying_cost", 1)) * self.config["general"].get("number_of_ships", 1)

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
        co2_release_cost = co2_released_in_factory * self.config["general"].get("m3_to_tons", 0.9) * self.kpis.get("co2_release_cost_per_ton", 1)

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
