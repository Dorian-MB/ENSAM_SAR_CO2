from colorama import Fore
import numpy as np
from pymoo.core.variable import Real, Integer, Choice
from pymoo.core.problem import ElementwiseProblem

from optimizer.utils import (
    ConfigBuilderFromSolution,
    calculate_performance_metrics,
    Normalizer,
)
from optimizer.boundaries import ConfigBoundaries
from eco2_normandy.simulation import Simulation
from eco2_normandy.logger import Logger


class SimulationProblem(ElementwiseProblem):
    """Problem definition for the simulation optimization using a genetic algorithm."""

    def __init__(
        self,
        base_config: dict,
        boundaries: ConfigBoundaries = None,
        kpis_list: list | None = None,
        logger=None,
        caps_steps: int = 1000,
        elementwise_runner=None,
    ) -> None:
        """Initialize the simulation problem, as a pymoo ElementwiseProblem with discrete variables."""
        self.log = logger or Logger()
        self.base_config = base_config
        self.boundaries = boundaries or ConfigBoundaries()
        self.max_ships = self.boundaries.max_num_ships
        self.caps_steps = caps_steps
        # default weights from Normalizer for consistency across modules
        self.weights = Normalizer().metrics_weight
        self.metrics_keys = Normalizer().metrics_keys
        self.cfg_builder = ConfigBuilderFromSolution(base_config, self.boundaries)
        self.kpis_list = kpis_list

        # Define variables with proper types and automatic steps
        self.variables = {}
        self.var_names = []

        # Core variables with discrete types
        self.variables["num_storages"] = Integer(bounds=(1, self.boundaries.max_num_storages))
        self.variables["use_Bergen"] = Choice(options=[0, 1])
        self.variables["use_Rotterdam"] = Choice(options=[0, 1])
        self.variables["num_ship"] = Integer(bounds=(1, self.boundaries.max_num_ships))
        self.variables["ship_speed"] = Integer(
            bounds=(self.boundaries.ship_speed["min"], self.boundaries.ship_speed["max"])
        )
        self.variables["number_of_tanks"] = Integer(
            bounds=(self.boundaries.factory_tanks["min"], self.boundaries.factory_tanks["max"])
        )

        # Variables with automatic steps
        self.variables["ship_capacity"] = Integer(
            bounds=(
                self.boundaries.ship_capacity["min"] // caps_steps,
                self.boundaries.ship_capacity["max"] // caps_steps,
            )
        )
        self.variables["storage_caps"] = Integer(
            bounds=(
                self.boundaries.storage_caps["min"] // caps_steps,
                self.boundaries.storage_caps["max"] // caps_steps,
            )
        )

        # Variables per ship (destinations) with explicit choices
        for i in range(self.boundaries.max_num_ships):
            self.variables[f"init{i + 1}_destination"] = Choice(options=[0, 1, 2])  # Le Havre, Rotterdam, Bergen
            self.variables[f"fixed{i + 1}_storage_destination"] = Choice(options=[0, 1])  # Rotterdam, Bergen

        # Store variable names in order for array conversion
        self.var_names = list(self.variables.keys())
        n_var = len(self.var_names)
        n_obj = 4

        # Create bounds arrays from variables (for ElementwiseProblem compatibility)
        xl = []
        xu = []
        for var_name in self.var_names:
            var = self.variables[var_name]
            if isinstance(var, (Integer, Real)):
                xl.append(var.bounds[0])
                xu.append(var.bounds[1])
            elif isinstance(var, Choice):
                xl.append(0)
                xu.append(len(var.options) - 1)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=np.array(xl, dtype=int),
            xu=np.array(xu, dtype=int),
            elementwise=True,
            elementwise_runner=elementwise_runner,
        )

    def _array_to_dict(self, x: np.ndarray) -> dict:
        """Convert array solution to dictionary, applying steps and choice mapping"""
        x_dict = {}
        for name, val in zip(self.var_names, x, strict=True):
            var = self.variables[name]

            if isinstance(var, Integer):
                # Apply steps for capacity variables
                if name in ["ship_capacity", "storage_caps"]:
                    x_dict[name] = int(val) * self.caps_steps
                else:
                    x_dict[name] = int(val)
            elif isinstance(var, Choice):
                # Map choice index to actual option value
                x_dict[name] = var.options[int(val)]
            elif isinstance(var, Real):
                x_dict[name] = float(val)
            else:
                x_dict[name] = int(val)

        return x_dict

    def _dict_to_array(self, x_dict: dict) -> np.ndarray:
        """Convert dictionary solution back to array, reversing steps and choice mapping"""
        x_array = []
        for name in self.var_names:
            var = self.variables[name]
            val = x_dict[name]

            if isinstance(var, Integer):
                # Reverse steps for capacity variables
                if name in ["ship_capacity", "storage_caps"]:
                    x_array.append(val // self.caps_steps)
                else:
                    x_array.append(int(val))
            elif isinstance(var, Choice):
                # Map option value back to choice index
                x_array.append(var.options.index(val))
            elif isinstance(var, Real):
                x_array.append(float(val))
            else:
                x_array.append(int(val))

        return np.array(x_array)

    def _run_simulation(self, cfg: dict) -> Simulation:
        """Run the simulation with animation."""
        sim = Simulation(config=cfg, verbose=False)
        try:
            sim.run()
        except Exception as e:
            self.log.error(Fore.RED + f"Simulation failed, config: {cfg}" + Fore.RESET)
            raise e
        return sim

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        # Convert array to dictionary (steps are applied automatically)
        x_dict = self._array_to_dict(x)

        # Build simulation config directly from variable dict
        cfg = self.cfg_builder.build(sol=x_dict)
        sim = self._run_simulation(cfg=cfg)
        metrics = calculate_performance_metrics(cfg=cfg, sim=sim, metrics_keys=self.metrics_keys)

        # Assign objectives
        objectives = [metrics[k].iloc[0] for k in self.metrics_keys]
        self.kpis_list.append(metrics)
        out["F"] = objectives

