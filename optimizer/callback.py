from ortools.sat.python.cp_model import CpSolverSolutionCallback
from colorama import Fore
import pandas as pd

from eco2_normandy.simulation import Simulation
from KPIS import Kpis
from KPIS.utils import compute_dynamic_bounds, normalize_dynamic
from optimizer.utils import ParetoFront, surrogate_metrics, ConfigBuilderFromSolution, calculate_performance_metrics
from eco2_normandy.logger import Logger
                                                         
metrics_keys = ["cost", "wasted_production_over_time", "waiting_time", "underfill_rate"]
metrics_weight = [20, 20, 15, 30]

class SimCallback(CpSolverSolutionCallback):
    def __init__(self, variables, 
                 base_config:dict, 
                 max_evals:int=50, 
                 verbose:int=True, 
                 metrics_keys:list[str]=metrics_keys, 
                 metrics_weight:list[int]=metrics_weight,
                 logger=None, 
                 ):
        super().__init__()
        self.metrics_keys = metrics_keys
        self.vars = variables
        self.log = logger or Logger()
        self.base_config = base_config
        self.cfg_builder = ConfigBuilderFromSolution(base_config)
        self.max_evals = max_evals
        self.evals = 1
        self.solutions_tested = 0
        self.verbose = verbose
        self.weights = {k:w for k,w in zip(metrics_keys, metrics_weight)} # poids des KPIs
        self.pareto_front = ParetoFront(self.metrics_keys) 
        self.surrogate_front = ParetoFront() 
        self.kpis_list = []
        self.surrogate_met = []

        self.norm_metrics = pd.DataFrame(columns=self.metrics_keys)
        self.raw_metrics = pd.DataFrame(columns=self.metrics_keys)
        self.solutions = pd.DataFrame(columns=[var.name for var in self.vars])
        self.raw_metrics.index.name = "evals"
        self.solutions.index.name = "evals"

    def _compute_score(self, row:pd.Series) -> float:
        return sum(self.weights[k] * row[k] for k in self.weights)

    def _compute_df_score(self, df:pd.DataFrame)-> pd.DataFrame:
        return df.apply(self._compute_score, axis=1)

    def get_config_from_solution(self, sol):
        return self.cfg_builder.build(sol)

    def run_simulation(self, cfg):
        try :
            if self.verbose < 2: sim = Simulation(config=cfg, verbose=False)
            if self.verbose == 2: sim = Simulation(config=cfg, verbose=True)
            sim.run()
        except Exception as e:
            if self.verbose > 2: self.log.warning(Fore.RED + f"Simulation échouée pour la solution {self.evals}. {e}\n")
            if self.verbose >= 2: self.log.warning(f"{cfg}"+ Fore.RESET)
            return None
        return sim

    def calculate_performance_metrics(self, cfg, sim):
        return calculate_performance_metrics(cfg, sim, metrics_keys=self.metrics_keys)
    
    def dynamic_normalize_metrics(self, metrics:dict, dynamic_bounds:list=None, clip:bool=True)-> dict[str, float]:
        """Normalise les métriques pour les mettre dans un intervalle dynamiquement."""
        if dynamic_bounds is None:
            dynamic_bounds = self.dynamic_bounds
        return normalize_dynamic(metrics, dynamic_bounds, clip)

    @property
    def dynamic_bounds(self):
        return compute_dynamic_bounds(self.kpis_list)

    @property
    def surrogate_dynamic_bounds(self):
        return compute_dynamic_bounds(self.surrogate_met)

    def _add_metrics_to_front(self, metrics:list|dict, front:ParetoFront, front_name:str="")->bool:
        """Met à jour le front de Pareto avec les nouvelles métriques.

        Args:
            metrics (list | dict): Les métriques à ajouter.
            front (ParetoFront): Le front de Pareto à mettre à jour.
            msg (str, optional): Un message à afficher en cas de domination. Defaults to "".

        Returns:
            bool: booléen indiquant si la solution a été ajoutée au front de Pareto.
        """
        if isinstance(metrics, list):
            metrics = metrics[-1]

        # Si cette solution est dominée par le front de pareto, on skip
        if front.is_dominated(metrics):
            msg = f"Solution {self.solutions_tested} dominée par le {front_name}, on passe a la suivante."
            if self.verbose > 2: self.log.info(Fore.YELLOW + msg + Fore.RESET)
            return False # passe à la solution suivante
        # Sinon on l’ajoute au front de pareto et on simule
        front.add(metrics, self.evals)
        return True

    def on_solution_callback(self):
        self.solutions_tested += 1

        # 2.1 Limiter le nombre d'évaluations
        if self.evals > self.max_evals:
            if self.verbose >= 1: self.log.info(Fore.RED + f"=== Limite d'évaluations atteinte ({self.max_evals}) ===" + Fore.RESET)
            self.StopSearch()
            return

        # 2.2 Extraire la solution CP-SAT
        sol = {v.Name(): self.Value(v) for v in self.vars}

        sur_met = surrogate_metrics(sol, self.base_config)
        self.surrogate_met.append(sur_met)
        if not self._add_metrics_to_front(self.surrogate_met, self.surrogate_front, front_name="surrogate_front"):
            return

        if self.verbose > 2:
            self.log.info(f"{Fore.LIGHTMAGENTA_EX}# Éval n°{self.evals}/{self.max_evals}: Total of {self.solutions_tested} sol tested: {Fore.RESET}\n"
                f"\t-ship:{sol['num_ship']}-ship-capa:{sol['ship_capacity']}-speed:{sol['ship_speed']}\n"
                f"\t-storge:{sol['num_storages']}")

        cfg = self.get_config_from_solution(sol)
        sim = self.run_simulation(cfg)
        if sim is None: # if simulation failed, skip this solution, and continue to the next one
            return
        metrics = self.calculate_performance_metrics(cfg, sim)
        self.kpis_list.append(metrics)
        if not self._add_metrics_to_front(self.kpis_list, self.pareto_front, front_name="pareto_front"):
            return

        if self.verbose > 2:
            self.log.info(Fore.GREEN +
                  f"→ Coût total combiné = {self.kpis_list[-1]['cost']:,.0f} €\n"
                  f"\t→ production gaspillée = {self.kpis_list[-1]['wasted_production_over_time']:,.0f} m^3\n"
                  f"\t→ Temps d'attente total = {self.kpis_list[-1]['waiting_time']:,.0f} s\n"
                  f"\t→ Taux de remplissage de l'usine = {self.kpis_list[-1]['underfill_rate']*100:.2f}%\n"
                  + Fore.RESET)

        self.norm_metrics.loc[self.evals] = metrics
        self.raw_metrics.loc[self.evals] = self.kpis_list[-1]
        self.solutions.loc[self.evals] = sol
        self.evals += 1

    def _normalize_metrics(self):
        """ Re-normalize kpis metrics after run is done, to use same updated dynamic bounds on all solutions."""
        f = lambda row: self.dynamic_normalize_metrics(row)
        raw_data = self.raw_metrics.drop(["score"], axis=1, errors='ignore')
        normalized_series = raw_data.apply(f, axis=1)
        normalized_df = pd.DataFrame(normalized_series.tolist(), index=normalized_series.index)
        self.norm_metrics = normalized_df

    def _compute_raw_scores(self):
        """Retourne les scores bruts finaux sous forme de DataFrame."""
        if self.raw_metrics.empty:
            self.log.info(Fore.YELLOW + "Aucun score brut calculé, DataFrame vide." + Fore.RESET)
            return
        if "score" in self.raw_metrics.columns:
            return
        self.raw_metrics["score"] = self._compute_df_score(self.norm_metrics)
        median = self.raw_metrics["score"].median()
        # suppression des lignes avec un score > 10 * median 
        filtre = self.raw_metrics["score"] <= 10 * median
        self.raw_metrics = self.raw_metrics[filtre]
        self.norm_metrics = self.norm_metrics[self.norm_metrics.index.isin(self.raw_metrics.index)]
        self.solutions = self.solutions[self.solutions.index.isin(self.raw_metrics.index)]
    
    def best_raw_score(self):
        """Retourne le meilleur score brut."""
        if "score" not in self.raw_metrics.columns:
            self.log.info(Fore.YELLOW + "Aucun score calculé." + Fore.RESET)
            return None
        return self.raw_metrics.loc[self.raw_metrics["score"].idxmin()]

        



