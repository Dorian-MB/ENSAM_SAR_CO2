import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from pymoo.indicators.hv import HV

# from pymoo.indicators.igd import IGD
# from pymoo.indicators.gd import GD
# from pymoo.util.normalization import normalize
import warnings

warnings.filterwarnings("ignore")

plt.style.use("ggplot")


class NSGA3HistoryAnalyzer:
    """Classe pour analyser l'historique d'un algorithme NSGA3 avec 4 objectifs"""

    def __init__(self, res, problem=None):
        """
        Initialisation de l'analyseur

        Args:
            res: Résultat de minimize() avec save_history=True
            problem: Problème optimisé (optionnel, pour le front de Pareto théorique)
        """
        self.res = res
        self.problem = problem
        self.n_obj = 4  # Pour 4 objectifs
        self.history = res.history if hasattr(res, "history") else []

    # ========================================================================
    # 1. ANALYSE DE LA CONVERGENCE
    # ========================================================================

    def analyze_convergence(self, ref_point=None, normalize_data=True):
        """
        Analyse complète de la convergence avec plusieurs métriques

        Args:
            ref_point: Point de référence pour l'hypervolume (par défaut: nadir * 1.1)
            normalize_data: Si True, normalise les données entre 0 et 1
        """
        metrics = {
            "generation": [],
            "n_solutions": [],
            "hypervolume": [],
            "spread": [],
            "spacing": [],
            "n_dominated": [],
            "mean_cv": [],  # Violation moyenne des contraintes
        }

        # Déterminer le point de référence si non fourni
        if ref_point is None:
            all_F = []
            for algo in self.history:
                if algo.opt is not None:
                    all_F.append(algo.opt.get("F"))
            if all_F:
                all_F = np.vstack(all_F)
                ref_point = np.max(all_F, axis=0) * 1.1
            else:
                ref_point = np.ones(self.n_obj)

        # print(f"Point de référence pour HV: {ref_point}")

        # Calculer les métriques pour chaque génération
        for i, algo in enumerate(self.history):
            metrics["generation"].append(i + 1)

            # Population optimale (front de Pareto courant)
            if algo.opt is not None and len(algo.opt) > 0:
                F_opt = algo.opt.get("F")

                # Normalisation si demandée
                if normalize_data and len(F_opt) > 1:
                    F_norm = self._normalize_objectives(F_opt)
                else:
                    F_norm = F_opt

                metrics["n_solutions"].append(len(F_opt))

                # Hypervolume (métrique principale)
                if len(F_norm) > 0:
                    hv = HV(ref_point=ref_point if not normalize_data else np.ones(self.n_obj))
                    metrics["hypervolume"].append(hv(F_norm))
                else:
                    metrics["hypervolume"].append(0)

                # Spread (diversité)
                metrics["spread"].append(self._calculate_spread(F_norm))

                # Spacing (uniformité)
                metrics["spacing"].append(self._calculate_spacing(F_norm))

                # Solutions dominées dans la population totale
                pop_F = algo.pop.get("F")
                n_dominated = len(pop_F) - len(F_opt)
                metrics["n_dominated"].append(n_dominated)

                # Violation des contraintes
                if algo.pop.get("CV") is not None:
                    cv = algo.pop.get("CV")
                    metrics["mean_cv"].append(np.mean(cv))
                else:
                    metrics["mean_cv"].append(0)
            else:
                # Pas de solutions optimales
                metrics["n_solutions"].append(0)
                metrics["hypervolume"].append(0)
                metrics["spread"].append(0)
                metrics["spacing"].append(0)
                metrics["n_dominated"].append(len(algo.pop))
                metrics["mean_cv"].append(0)

        return pd.DataFrame(metrics)

    def plot_convergence(self, metrics_df=None, figsize=(15, 10)):
        """Visualisation de la convergence avec plusieurs métriques"""
        if metrics_df is None:
            metrics_df = self.analyze_convergence()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Analyse de Convergence NSGA3 - 4 Objectifs", fontsize=16)

        # 1. Hypervolume
        ax = axes[0, 0]
        ax.plot(metrics_df["generation"], metrics_df["hypervolume"], "b-", linewidth=2)
        ax.scatter(metrics_df["generation"], metrics_df["hypervolume"], c="blue", s=30)
        ax.set_xlabel("Génération")
        ax.set_ylabel("Hypervolume")
        ax.set_title("Convergence (Hypervolume)")
        ax.grid(True, alpha=0.3)

        # 2. Nombre de solutions non-dominées
        ax = axes[0, 1]
        ax.plot(metrics_df["generation"], metrics_df["n_solutions"], "g-", linewidth=2)
        ax.scatter(metrics_df["generation"], metrics_df["n_solutions"], c="green", s=30)
        ax.set_xlabel("Génération")
        ax.set_ylabel("Nombre de solutions")
        ax.set_title("Taille du Front de Pareto")
        ax.grid(True, alpha=0.3)

        # 3. Spread (Diversité)
        ax = axes[0, 2]
        ax.plot(metrics_df["generation"], metrics_df["spread"], "r-", linewidth=2)
        ax.scatter(metrics_df["generation"], metrics_df["spread"], c="red", s=30)
        ax.set_xlabel("Génération")
        ax.set_ylabel("Spread")
        ax.set_title("Diversité des Solutions")
        ax.grid(True, alpha=0.3)

        # 4. Spacing (Uniformité)
        ax = axes[1, 0]
        ax.plot(metrics_df["generation"], metrics_df["spacing"], "orange", linewidth=2)
        ax.scatter(metrics_df["generation"], metrics_df["spacing"], c="orange", s=30)
        ax.set_xlabel("Génération")
        ax.set_ylabel("Spacing")
        ax.set_title("Uniformité de Distribution")
        ax.grid(True, alpha=0.3)

        # 5. Solutions dominées
        ax = axes[1, 1]
        ax.plot(metrics_df["generation"], metrics_df["n_dominated"], "purple", linewidth=2)
        ax.scatter(metrics_df["generation"], metrics_df["n_dominated"], c="purple", s=30)
        ax.set_xlabel("Génération")
        ax.set_ylabel("Solutions dominées")
        ax.set_title("Solutions Dominées dans la Population")
        ax.grid(True, alpha=0.3)

        # 6. Violation des contraintes
        ax = axes[1, 2]
        ax.plot(metrics_df["generation"], metrics_df["mean_cv"], "brown", linewidth=2)
        ax.scatter(metrics_df["generation"], metrics_df["mean_cv"], c="brown", s=30)
        ax.set_xlabel("Génération")
        ax.set_ylabel("Violation moyenne")
        ax.set_title("Violation des Contraintes")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ========================================================================
    # 2. COMPARAISON D'ALGORITHMES
    # ========================================================================

    @staticmethod
    def compare_algorithms(results_dict, ref_point=None, normalize_data=True):
        """
        Compare plusieurs algorithmes NSGA3

        Args:
            results_dict: Dict {'nom_algo': res, ...}
            ref_point: Point de référence pour HV
            normalize_data: Normaliser les objectifs

        Returns:
            DataFrame avec comparaison des métriques
        """
        comparison = {}

        for algo_name, res in results_dict.items():
            analyzer = NSGA3HistoryAnalyzer(res)
            metrics = analyzer.analyze_convergence(ref_point, normalize_data)

            # Métriques finales
            final_metrics = metrics.iloc[-1]

            # Métriques de convergence
            hv_improvement = metrics["hypervolume"].iloc[-1] - metrics["hypervolume"].iloc[0]
            convergence_speed = analyzer._calculate_convergence_speed(metrics["hypervolume"].values)

            comparison[algo_name] = {
                "final_hypervolume": final_metrics["hypervolume"],
                "final_n_solutions": final_metrics["n_solutions"],
                "final_spread": final_metrics["spread"],
                "final_spacing": final_metrics["spacing"],
                "hv_improvement": hv_improvement,
                "convergence_speed": convergence_speed,
                "n_generations": len(metrics),
                "final_cv": final_metrics["mean_cv"],
            }

        return pd.DataFrame(comparison).T

    @staticmethod
    def plot_algorithms_comparison(results_dict, figsize=(15, 5)):
        """Visualisation de la comparaison entre algorithmes"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("Comparaison des Algorithmes NSGA3", fontsize=16)

        colors = plt.cm.Set3(np.linspace(0, 1, len(results_dict)))

        for idx, (algo_name, res) in enumerate(results_dict.items()):
            analyzer = NSGA3HistoryAnalyzer(res)
            metrics = analyzer.analyze_convergence()
            color = colors[idx]

            # Hypervolume
            axes[0].plot(metrics["generation"], metrics["hypervolume"], label=algo_name, color=color, linewidth=2)

            # Nombre de solutions
            axes[1].plot(metrics["generation"], metrics["n_solutions"], label=algo_name, color=color, linewidth=2)

            # Spread
            axes[2].plot(metrics["generation"], metrics["spread"], label=algo_name, color=color, linewidth=2)

        axes[0].set_title("Hypervolume")
        axes[0].set_xlabel("Génération")
        axes[0].set_ylabel("HV")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Taille du Front de Pareto")
        axes[1].set_xlabel("Génération")
        axes[1].set_ylabel("Nombre")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_title("Diversité (Spread)")
        axes[2].set_xlabel("Génération")
        axes[2].set_ylabel("Spread")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ========================================================================
    # 3. DÉTECTION DE STAGNATION
    # ========================================================================

    def detect_stagnation(self, window_size=5, threshold=0.01):
        """
        Détecte les périodes de stagnation dans l'optimisation

        Args:
            window_size: Taille de la fenêtre pour calculer l'amélioration
            threshold: Seuil d'amélioration minimale (en %)

        Returns:
            Dict avec les périodes de stagnation détectées
        """
        metrics = self.analyze_convergence()
        hv_values = metrics["hypervolume"].values

        stagnation_periods = []
        improvements = []

        for i in range(window_size, len(hv_values)):
            window_start = i - window_size
            window_improvement = (hv_values[i] - hv_values[window_start]) / (hv_values[window_start] + 1e-10)
            improvements.append(window_improvement)

            if abs(window_improvement) < threshold:
                stagnation_periods.append(
                    {
                        "start": window_start + 1,
                        "end": i + 1,
                        "improvement": window_improvement,
                        "hv_start": hv_values[window_start],
                        "hv_end": hv_values[i],
                    }
                )

        # Fusionner les périodes adjacentes
        merged_periods = []
        if stagnation_periods:
            current = stagnation_periods[0]
            for period in stagnation_periods[1:]:
                if period["start"] <= current["end"] + 1:
                    current["end"] = period["end"]
                    current["hv_end"] = period["hv_end"]
                else:
                    merged_periods.append(current)
                    current = period
            merged_periods.append(current)

        return {
            "stagnation_periods": merged_periods,
            "improvements": improvements,
            "total_stagnation_gens": sum(p["end"] - p["start"] + 1 for p in merged_periods),
            "stagnation_ratio": sum(p["end"] - p["start"] + 1 for p in merged_periods) / len(hv_values),
        }

    def plot_stagnation_analysis(self, window_size=5, threshold=0.01, figsize=(15, 5)):
        """Visualise l'analyse de stagnation"""
        stagnation_info = self.detect_stagnation(window_size, threshold)
        metrics = self.analyze_convergence()

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("Analyse de Stagnation NSGA3", fontsize=16)

        # 1. Hypervolume avec zones de stagnation
        ax = axes[0]
        ax.plot(metrics["generation"], metrics["hypervolume"], "b-", linewidth=2)

        # Marquer les zones de stagnation
        for period in stagnation_info["stagnation_periods"]:
            ax.axvspan(period["start"], period["end"], alpha=0.3, color="red")

        ax.set_xlabel("Génération")
        ax.set_ylabel("Hypervolume")
        ax.set_title(f"Zones de Stagnation (seuil={threshold * 100:.1f}%)")
        ax.grid(True, alpha=0.3)

        # 2. Taux d'amélioration
        ax = axes[1]
        improvements = stagnation_info["improvements"]
        gens = list(range(window_size + 1, len(metrics) + 1))

        ax.plot(gens, improvements, "g-", linewidth=2)
        ax.axhline(y=threshold, color="r", linestyle="--", label=f"Seuil ({threshold * 100:.1f}%)")
        ax.axhline(y=-threshold, color="r", linestyle="--")
        ax.set_xlabel("Génération")
        ax.set_ylabel("Taux d'amélioration")
        ax.set_title(f"Amélioration sur {window_size} générations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Statistiques de stagnation
        ax = axes[2]
        ax.axis("off")

        stats_text = f"""Statistiques de Stagnation:
        
• Nombre de périodes: {len(stagnation_info["stagnation_periods"])}
• Générations en stagnation: {stagnation_info["total_stagnation_gens"]}
• Ratio de stagnation: {stagnation_info["stagnation_ratio"] * 100:.1f}%

Périodes détectées:"""

        for i, period in enumerate(stagnation_info["stagnation_periods"][:5]):  # Max 5 périodes
            stats_text += f"\n  {i + 1}. Gen {period['start']}-{period['end']} "
            stats_text += f"(Δ={period['improvement'] * 100:.2f}%)"

        ax.text(
            0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace"
        )

        plt.tight_layout()
        return fig

    # ========================================================================
    # 4. VISUALISATION DE L'ÉVOLUTION
    # ========================================================================

    def visualize_evolution(self, generations_to_show=None, figsize=(15, 10)):
        """
        Visualise l'évolution du front de Pareto pour 4 objectifs

        Args:
            generations_to_show: Liste des générations à afficher (par défaut: équidistantes)
        """
        if generations_to_show is None:
            n_gens = len(self.history)
            # Montrer 6 générations équidistantes
            generations_to_show = np.linspace(0, n_gens - 1, min(6, n_gens), dtype=int)

        n_plots = len(generations_to_show)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle("Évolution du Front de Pareto (4 Objectifs)", fontsize=16)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        # Pour chaque génération sélectionnée
        for idx, gen_idx in enumerate(generations_to_show):
            ax = axes[idx]
            algo = self.history[gen_idx]

            if algo.opt is not None and len(algo.opt) > 0:
                F = algo.opt.get("F")

                # Projection 2D des 4 objectifs (parallel coordinates style)
                self._plot_4d_projection(F, ax, f"Génération {gen_idx + 1}")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Génération {gen_idx + 1}\n(Pas de solutions)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        # Cacher les axes non utilisés
        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        return fig

    def create_evolution_animation(self, filename="evolution.gif", fps=2):
        """
        Crée une animation GIF de l'évolution du front de Pareto

        Args:
            filename: Nom du fichier de sortie
            fps: Images par seconde
        """
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def update(frame):
            ax1.clear()
            ax2.clear()

            algo = self.history[frame]

            # Titre principal
            fig.suptitle(f"NSGA3 - Génération {frame + 1}/{len(self.history)}", fontsize=14)

            if algo.opt is not None and len(algo.opt) > 0:
                F = algo.opt.get("F")

                # Graphique 1: Parallel coordinates
                self._plot_parallel_coordinates(F, ax1, "Coordonnées Parallèles")

                # Graphique 2: Matrice de corrélation
                self._plot_correlation_matrix(F, ax2, "Matrice de Corrélation")
            else:
                ax1.text(0.5, 0.5, "Pas de solutions", transform=ax1.transAxes, ha="center", va="center")
                ax2.text(0.5, 0.5, "Pas de solutions", transform=ax2.transAxes, ha="center", va="center")

        anim = FuncAnimation(fig, update, frames=len(self.history), interval=1000 / fps, repeat=True)

        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"Animation sauvegardée: {filename}")

        return anim

    # ========================================================================
    # MÉTHODES UTILITAIRES
    # ========================================================================

    def _normalize_objectives(self, F):
        """Normalise les objectifs entre 0 et 1"""
        if len(F) == 0:
            return F

        min_vals = np.min(F, axis=0)
        max_vals = np.max(F, axis=0)
        range_vals = max_vals - min_vals

        # Éviter la division par zéro
        range_vals[range_vals < 1e-10] = 1.0

        return (F - min_vals) / range_vals

    def _calculate_spread(self, F):
        """Calcule le spread (diversité) des solutions"""
        if len(F) <= 1:
            return 0

        # Distance maximale entre chaque paire de points
        distances = []
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                dist = np.linalg.norm(F[i] - F[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0

    def _calculate_spacing(self, F):
        """Calcule le spacing (uniformité) des solutions"""
        if len(F) <= 1:
            return 0

        # Distance au plus proche voisin pour chaque point
        min_distances = []
        for i in range(len(F)):
            distances = []
            for j in range(len(F)):
                if i != j:
                    dist = np.linalg.norm(F[i] - F[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))

        if min_distances:
            mean_dist = np.mean(min_distances)
            std_dist = np.std(min_distances)
            return std_dist / (mean_dist + 1e-10)
        return 0

    def _calculate_convergence_speed(self, hv_values):
        """Calcule la vitesse de convergence"""
        if len(hv_values) <= 1:
            return 0

        # Fit polynomial pour smooth
        x = np.arange(len(hv_values))
        coeffs = np.polyfit(x, hv_values, min(3, len(hv_values) - 1))
        poly = np.poly1d(coeffs)

        # Dérivée moyenne
        derivative = np.gradient(poly(x))
        return np.mean(derivative)

    def _plot_4d_projection(self, F, ax, title):
        """Projette 4 objectifs en 2D pour visualisation"""
        # Utiliser les 2 premiers objectifs principaux
        from sklearn.decomposition import PCA

        if len(F) > 2:
            pca = PCA(n_components=2)
            F_2d = pca.fit_transform(F)

            scatter = ax.scatter(F_2d[:, 0], F_2d[:, 1], c=np.arange(len(F)), cmap="viridis", s=50, alpha=0.7)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
            plt.colorbar(scatter, ax=ax, label="Index")
        else:
            # Si moins de 3 solutions, afficher directement
            ax.scatter(F[:, 0], F[:, 1], s=50, alpha=0.7)
            ax.set_xlabel("Objectif 1")
            ax.set_ylabel("Objectif 2")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _plot_parallel_coordinates(self, F, ax, title):
        """Trace les coordonnées parallèles pour 4 objectifs"""
        n_solutions = len(F)
        n_obj = F.shape[1]

        # Normaliser pour l'affichage
        F_norm = self._normalize_objectives(F)

        x = np.arange(n_obj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_solutions))

        for i, solution in enumerate(F_norm):
            ax.plot(x, solution, "o-", color=colors[i], alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Obj {i + 1}" for i in range(n_obj)])
        ax.set_ylabel("Valeur normalisée")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    def _plot_correlation_matrix(self, F, ax, title):
        """Affiche la matrice de corrélation entre objectifs"""
        if len(F) < 2:
            ax.text(0.5, 0.5, "Pas assez de données", transform=ax.transAxes, ha="center", va="center")
            return

        corr_matrix = np.corrcoef(F.T)

        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Labels
        n_obj = F.shape[1]
        ax.set_xticks(np.arange(n_obj))
        ax.set_yticks(np.arange(n_obj))
        ax.set_xticklabels([f"O{i + 1}" for i in range(n_obj)])
        ax.set_yticklabels([f"O{i + 1}" for i in range(n_obj)])

        # Afficher les valeurs
        for i in range(n_obj):
            for j in range(n_obj):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Corrélation")


# ========================================================================
# EXEMPLE D'UTILISATION
# ========================================================================


def example_usage():
    """Exemple d'utilisation de l'analyseur"""

    # Supposons que vous avez déjà exécuté votre optimisation:
    # res = minimize(problem, algorithm, termination, save_history=True)

    # Créer l'analyseur
    # analyzer = NSGA3HistoryAnalyzer(res)

    # 1. Analyse de convergence
    # metrics_df = analyzer.analyze_convergence()
    # print("\nMétriques de convergence:")
    # print(metrics_df.tail())

    # fig = analyzer.plot_convergence()
    # plt.show()

    # 2. Détection de stagnation
    # stagnation = analyzer.detect_stagnation(window_size=5, threshold=0.01)
    # print(f"\nRatio de stagnation: {stagnation['stagnation_ratio']*100:.1f}%")

    # fig = analyzer.plot_stagnation_analysis()
    # plt.show()

    # 3. Visualisation de l'évolution
    # fig = analyzer.visualize_evolution()
    # plt.show()

    # 4. Créer une animation (optionnel)
    # analyzer.create_evolution_animation('nsga3_evolution.gif', fps=2)

    # 5. Comparaison d'algorithmes (si plusieurs résultats)
    # results = {
    #     'NSGA3_config1': res1,
    #     'NSGA3_config2': res2,
    # }
    # comparison_df = NSGA3HistoryAnalyzer.compare_algorithms(results)
    # print("\nComparaison des algorithmes:")
    # print(comparison_df)

    print("Exemple de code pour utiliser l'analyseur d'historique NSGA3")


if __name__ == "__main__":
    example_usage()
