import os

import numpy as np
from multiprocessing import Pool
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer, Choice
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.problem import StarmapParallelization


class MixedVariableSampling(Sampling):
    """
    Custom sampling that understands variable types (Integer, Choice, Real).
    Generates samples respecting the semantic of each variable type.
    """

    def __init__(self, problem: Problem = None):
        super().__init__()
        self.problem = problem

    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        if self.problem is None:
            self.problem = problem
        # Generate samples for each variable type
        X = np.zeros((n_samples, problem.n_var), dtype=int)

        for i, var_name in enumerate(problem.var_names):
            var = problem.variables[var_name]

            if isinstance(var, Integer):
                # Integer variables with bounds
                X[:, i] = np.random.randint(var.bounds[0], var.bounds[1] + 1, size=n_samples)
            elif isinstance(var, Choice):
                # Choice variables - select from options
                X[:, i] = np.random.choice(range(len(var.options)), size=n_samples)
            elif isinstance(var, Real):
                # Real variables (if any)
                X[:, i] = np.random.uniform(var.bounds[0], var.bounds[1], size=n_samples).astype(int)

        return X


class ShipConsistencyRepair(Repair):
    """
    Simplified repair operator leveraging ship._pick_new_destination() behavior.

    Key insight: When no fixed_storage_destination is set in the ship config,
    ship._pick_new_destination() uses random.choice(self.storages), which automatically
    handles storage availability. This allows for much simpler repair logic.
    """

    def __init__(self, max_ships: int, problem: Problem = None):
        super().__init__()
        self.max_ships = max_ships
        self.problem = problem

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Repair solutions with simplified logic:
        1. Force unused ships to default values (main efficiency gain)
        2. Set storage flags based on what's available in the scenario
        3. Minimal fixed destination repair (ship handles the rest automatically)

        Args:
            problem: The optimization problem
            X: Array of solutions (n_solutions x n_variables)

        Returns:
            X: Repaired solutions
        """
        # Store problem reference for variable name mapping
        if self.problem is None:
            self.problem = problem

        for i, x in enumerate(X):
            # Convert to dict for easier manipulation
            x_dict = problem._array_to_dict(x)
            num_ships = x_dict["num_ship"]
            num_storages = x_dict["num_storages"]

            # 1. Force unused ships to default values (main repair benefit)
            for ship_idx in range(num_ships, self.max_ships):
                x_dict[f"init{ship_idx + 1}_destination"] = 0  # Le Havre
                x_dict[f"fixed{ship_idx + 1}_storage_destination"] = 0  # Default

            # 2. Storage use fixed: not twice the same; and ensure at least one is used; and not more than available
            if num_storages == 2:
                x_dict["use_Bergen"] = 1
                x_dict["use_Rotterdam"] = 1
            elif not x_dict["use_Bergen"] and not x_dict["use_Rotterdam"]:
                x_dict["use_Bergen"] = 0
                x_dict["use_Rotterdam"] = 1
            elif num_storages == 1 and x_dict["use_Bergen"] and x_dict["use_Rotterdam"]:
                x_dict["use_Bergen"] = 0
                x_dict["use_Rotterdam"] = 1

            has_bergen = x_dict["use_Bergen"] == 1
            has_rotterdam = x_dict["use_Rotterdam"] == 1

            # 3. Simple fixed destination repair - only fix obvious conflicts
            # map_ship_fixed_storage_destination = {0: "Rotterdam", 1: "Bergen"}
            # map_ship_initial_destination = {0: "Le Havre", 1: "Rotterdam", 2: "Bergen"}
            for ship_idx in range(num_ships):
                current_fixed = x_dict[f"fixed{ship_idx + 1}_storage_destination"]
                current_init = x_dict[f"init{ship_idx + 1}_destination"]

                # Only repair if the chosen storage is completely unavailable
                if current_fixed == 1 and not has_bergen:  # Wants Bergen but unavailable
                    x_dict[f"fixed{ship_idx + 1}_storage_destination"] = 0  # → Rotterdam
                elif current_fixed == 0 and not has_rotterdam:  # Wants Rotterdam but unavailable
                    x_dict[f"fixed{ship_idx + 1}_storage_destination"] = 1  # → Bergen

                # If the initial destination is unavailable, set it to the factory
                if current_init == 2 and not has_bergen:  # Wants Bergen but unavailable
                    x_dict[f"init{ship_idx + 1}_destination"] = 0
                elif current_init == 1 and not has_rotterdam:  # Wants Rotterdam but unavailable
                    x_dict[f"init{ship_idx + 1}_destination"] = 0

            # Convert back to array
            X[i] = problem._dict_to_array(x_dict)

        return X


class SerializableStarmapRunner(StarmapParallelization):  # heritage pas obligatoire
    def __init__(self, n_processes: int | None = None) -> None:
        self.n_processes = n_processes or os.cpu_count()
        self._pool = None

    def __call__(self, f, X):
        # Créer le pool à la demande
        if self._pool is None:
            self._pool = Pool(self.n_processes)
        return list(self._pool.starmap(f, [[x] for x in X]))

    def __getstate__(self):
        # Pour la sérialisation, ne pas inclure le pool
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def __setstate__(self, state):
        # Après désérialisation, le pool sera recréé à la demande
        self.__dict__.update(state)

    def close(self):
        if self._pool:
            self._pool.close()
            self._pool = None
