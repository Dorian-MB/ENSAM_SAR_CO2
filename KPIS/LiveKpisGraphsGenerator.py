import sys
import inspect
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

plt.style.use("ggplot")

from eco2_normandy.logger import Logger


class LiveKpisGraphsGenerator:
    """
    pipeline « Matplotlib → Agg canvas → buffer_rgba → Pygame Surface → scale → blit »
    On utilise une technique de blitting (via FigureCanvasAgg + canvas.blit)
    permetant de ne redessiner à l’écran que les portions qui changent dans Matplotlib.
    """

    def __init__(
        self,
        simulation_df,
        config,
        logger=None,
    ):
        self.config = config
        self.logger = logger or Logger()
        self.factory_name = self.config["factory"]["name"]
        self.storage_names = [s["name"] for s in self.config["storages"]]
        self.ship_names = [s["name"] for s in self.config["ships"]]
        self.figsize = (9, 4)
        self.kpis = config["KPIS"]
        self.num_period = self.config["general"]["num_period"]

        self.figs = dict()

        self.upload_data(simulation_df)
        self.graphs = {}
        self.canvas = {}
        self._init_graphs()

    def upload_data(self, df):
        self.dfs = df

    def on_resize(self, new_size, old_size):
        # mettre à jour la taille en pouces
        new_x, new_y = new_size
        old_x, old_y = old_size
        ratio_x, ratio_y = new_x / old_x, new_y / old_y
        # surf = pygame.transform.scale(surf, (self.WIDTH * ratio_w, self.HEIGHT * ratio_h)) # Pas tres efficasse => A changer
        figsize = ratio_x * self.figsize[0], ratio_y * self.figsize[1]

        # fermer les anciennes figures pour libérer la mémoire
        for fig, *_ in self.figs.values():
            plt.close(fig)
        self.figs.clear()
        self.canvas.clear()
        self.graphs.clear()

        # recréer tous les plots (init=True)
        self._init_graphs(figsize=figsize)

        # pré-dessiner une frame “live” avec les vraies données
        for key, updater in self.graphs.items():
            updater()  # ce call mettra à jour le fond + bodies

    def _init_canvas(self, nom):
        fig = self.figs[nom][0]
        canvas = FigureCanvas(fig)
        # IMPORTANT : dessiner la figure AVANT de capturer le fond
        canvas.draw()
        background = canvas.copy_from_bbox(fig.bbox)
        self.canvas[nom] = [canvas, background]

    def _init_graphs(self, figsize=None):
        """
        Initialise chaque graphe et stocke une fonction update(step) retournant (raw_rgba, (w,h)).
        """
        plot_factory_capacity_evolution = self._init_plot_factory_capacity_evolution(
            figsize
        )
        self._init_canvas(plot_factory_capacity_evolution)

        def update_capacity():
            fig, ax, (line, max_line) = self.figs[plot_factory_capacity_evolution]
            canvas, background = self.canvas[plot_factory_capacity_evolution]
            renderer = canvas.get_renderer()

            x, y, capa_max = self._get_data_factory_capacity_evolution()
            canvas.restore_region(background)
            line.set_data(x, y)
            ax.draw_artist(line)
            # x0, x1 = ax.get_xlim()
            # max_line.set_data([x0, x1], [capa_max, capa_max])
            # ax.draw_artist(max_line)
            canvas.blit(fig.bbox)
            raw = renderer.buffer_rgba()
            return raw, canvas.get_width_height()

        self.graphs["factory_capacity"] = update_capacity

        plot_factory_capacity_evolution_violin = (
            self._init_plot_factory_capacity_evolution_violin(figsize)
        )
        self._init_canvas(plot_factory_capacity_evolution_violin)

        def update_violin():
            fig, ax, (parts, box) = self.figs[plot_factory_capacity_evolution_violin]
            canvas, background = self.canvas[plot_factory_capacity_evolution_violin]
            renderer = canvas.get_renderer()
            data = self._get_data_factory_capacity_violin()
            canvas.restore_region(background)

            # Remove old violin bodies and boxplot artists
            for key, artists in parts.items():
                for art in np.atleast_1d(artists):
                    art.remove()
            for artist in (
                box["boxes"]
                + box["medians"]
                + box["whiskers"]
                + box["caps"]
                + box["fliers"]
                + box["means"]
            ):
                artist.remove()
            # Redraw new violin and boxplot
            parts_new = ax.violinplot(
                [data],
                widths=0.8,
                showmeans=False,
                showmedians=True,
                showextrema=True,
                quantiles=[0.25, 0.75],
            )
            for key, artists in parts_new.items():
                for art in np.atleast_1d(artists):
                    art.set_alpha(0.5)
                    art.set_color("red")
            box_new = ax.boxplot(
                [data],
                widths=0.1,
                whis=1.5,
                patch_artist=True,
                boxprops=dict(facecolor="salmon", edgecolor="red"),
                medianprops=dict(color="black"),
                flierprops=dict(
                    marker=".",
                    markerfacecolor="black",
                    markersize=4,
                    alpha=0.6,
                    c="red",
                ),
            )

            # Store new parts
            self.figs[plot_factory_capacity_evolution_violin][2] = (parts_new, box_new)
            # Draw artists
            for key, artists in parts_new.items():
                for art in np.atleast_1d(artists):
                    ax.draw_artist(art)
            for artist in (
                box_new["boxes"]
                + box_new["medians"]
                + box_new["whiskers"]
                + box_new["caps"]
                + box_new["fliers"]
            ):
                ax.draw_artist(artist)
            canvas.blit(fig.bbox)
            raw = renderer.buffer_rgba()
            return raw, canvas.get_width_height()

        self.graphs["factory_violin"] = update_violin

        plot_storage_capacity_comparison = self._init_plot_storage_capacity_comparison(
            figsize
        )
        self._init_canvas(plot_storage_capacity_comparison)

        def update_storage():
            fig, ax, (bars, annots) = self.figs[plot_storage_capacity_comparison]
            canvas, background = self.canvas[plot_storage_capacity_comparison]
            renderer = canvas.get_renderer()

            # get new counts
            values = self._get_data_storage_capacity_comparison()  # returns dict
            heights = list(values.values())
            canvas.restore_region(background)
            # update bars
            for bar, h, annot in zip(bars, heights, annots):
                bar.set_height(h)
                ax.draw_artist(bar)

                # reposition annotation at top of bar
                annot.set_text(str(h))
                xi = bar.get_x() + bar.get_width() / 2
                annot.xy = (xi, h)
                ax.draw_artist(annot)
            canvas.blit(fig.bbox)
            raw = renderer.buffer_rgba()
            return raw, canvas.get_width_height()

        self.graphs["storage_comparison"] = update_storage

        plot_wasted_production_over_time = self._init_plot_wasted_production_over_time(
            figsize
        )
        self._init_canvas(plot_wasted_production_over_time)

        def update_wasted_prouction():
            fig, ax, (line, max_) = self.figs[plot_wasted_production_over_time]
            canvas, background = self.canvas[plot_wasted_production_over_time]
            renderer = canvas.get_renderer()

            x, y = self._get_data_wasted_production_over_time()
            new_max = y.max()
            if new_max > max_:
                max_ = new_max * 1.2
                ax.set_ylim(0, max_)
                self.figs[plot_wasted_production_over_time][2][1] = max_
                self._set_ax_format(ax)  # Pas optimal de recalculer le background
                self._init_canvas(plot_wasted_production_over_time)
            canvas.restore_region(background)
            line.set_data(x, y)
            ax.draw_artist(line)

            canvas.blit(fig.bbox)
            raw = renderer.buffer_rgba()
            return raw, canvas.get_width_height()

        self.graphs["wasted_production"] = update_wasted_prouction

    def _set_ax_format(self, ax):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda val, pos: f"{int(val/1000)}k")
        )

    def _get_fig(self, nom, figsize=None):
        figsize = figsize or self.figsize
        self.figs[nom] = list(plt.subplots(figsize=figsize))
        return self.figs[nom]

    def _get_data_factory_capacity_evolution(self, init=False):
        # Extract storage capacity
        df = self.dfs[self.factory_name]
        x, y = df.index, df.capacity
        # Extract capacity_max for percentage calculation and plotting
        capa_max = df["capacity_max"].iloc[0]
        if init:
            return [0], [0], capa_max
        return x, y, capa_max

    def plot_factory_capacity_evolution(self):
        return self._init_plot_factory_capacity_evolution(init=False)

    def _init_plot_factory_capacity_evolution(self, figsize=None, init=True):
        nom = str(inspect.currentframe().f_code.co_name).replace("_init_", "")
        title = f"{self.factory_name} Capacity"

        x, y, capa_max = self._get_data_factory_capacity_evolution(init)

        fig, ax = self._get_fig(nom, figsize)
        line = ax.plot(
            x, y, marker=".", linestyle="-", linewidth=1, c="salmon", label="capacity"
        )  # Pourquoi ax.plot est dans une liste a 1 element ?
        # self.logger.info(line) # [<matplotlib.lines.Line2D object at 0x000001869533F070>]
        if type(line) == list:
            line = line[0]
        max_line = ax.axhline(
            capa_max, linestyle="--", linewidth=2, c="black", label="capacity max"
        )
        lines = [line, max_line]
        self.figs[nom].append(lines)
        ax.set_ylabel("Le Havre Capacity (T co2)")
        ax.set_xlabel("step")
        ax.annotate(
            "Max Capacity",
            xy=(self.num_period, capa_max),
            xytext=(-60, 5),
            textcoords="offset points",
            fontsize=9,
        )
        self._set_ax_format(ax)
        ax.set_xlim(-10, self.num_period * 1.1)
        ax.set_ylim(0, capa_max * 1.1)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return nom

    def _get_data_factory_capacity_violin(self, init=False):
        if init:
            return [0]
        return self.dfs[self.factory_name]["capacity"]

    def plot_factory_capacity_evolution_violin(self):
        return self._init_plot_factory_capacity_evolution_violin(init=False)

    def _init_plot_factory_capacity_evolution_violin(self, figsize=None, init=True):
        nom = str(inspect.currentframe().f_code.co_name).replace("_init_", "")
        fig, ax = self._get_fig(nom, figsize)
        capacity = self._get_data_factory_capacity_violin(init)
        title = f"{self.factory_name} Capacity: Violin Graph"
        parts = ax.violinplot(
            [capacity],
            widths=0.8,
            showmeans=False,
            showmedians=not init,
            showextrema=not init,
            quantiles=[0.25, 0.75] if init is False else None,
        )
        for pc in parts[
            "bodies"
        ]:  # contient les polygones correspondant à la forme du violon.
            pc.set_alpha(0.5)
            pc.set_facecolor("red")

            # Boxplot pour quartiles + outliers
        box = ax.boxplot(
            [capacity],
            widths=0.1,
            whis=1.5,  # (whiskers) partent de chaque extrémité de la boîte et s’étendent jusqu’à la valeur la plus extrême située à moins de 1,5 × IQR (IQR = Q3–Q1) de la boîte.
            patch_artist=True,
            boxprops=dict(facecolor="salmon", edgecolor="red"),
            medianprops=dict(color="black"),
            flierprops=dict(
                marker=".", markerfacecolor="black", markersize=4, alpha=0.6, c="red"
            ),
        )
        violon = [parts, box]
        self.figs[nom].append(violon)
        # Etiquettes
        ax.set_ylim(0, self.dfs[self.factory_name].capacity_max.iloc[0] * 1.1)
        self._set_ax_format(ax)
        ax.set_xticks([1])
        ax.set_xticklabels([title], ha="center")
        ax.set_title(title)
        ax.set_ylabel("Capacity")
        fig.tight_layout()
        return nom

    def _get_data_storage_capacity_comparison(self, init=False):
        df = self.dfs[self.factory_name]
        equal = df[df.capacity == df.capacity_max].shape[0]
        not_equal = df.shape[0] - equal
        if init:
            equal, not_equal = 0, 0
        return {"capacity_max == capacity": equal, "capacity_max > capacity": not_equal}

    def plot_storage_capacity_comparison(self):
        return self._init_plot_storage_capacity_comparison(init=False)

    def _init_plot_storage_capacity_comparison(self, figsize=None, init=True):
        nom = str(inspect.currentframe().f_code.co_name).replace("_init_", "")
        fig, ax = self._get_fig(nom, figsize)
        data = self._get_data_storage_capacity_comparison(init)
        labels, values = list(data.keys()), list(data.values())
        bar = ax.bar(labels, values, color=["lightblue", "salmon"])
        annots = []
        for label, val in data.items():
            annot = ax.annotate(
                str(val),
                xy=(label, val),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
            )
            annots.append(annot)

        ax.set_ylim(0, self.num_period)
        ax.set_ylabel("Total Hours")
        ax.set_xlabel("Condition")
        fig.tight_layout()
        bars = [bar, annots]
        self.figs[nom].append(bars)
        return nom

    def _get_data_wasted_production_over_time(self, init=False):
        if init:
            x = y = np.array([0])
            return x, y
        df = self.dfs[self.factory_name]
        wasted_production_over_time = df.wasted_production.cumsum()
        step = df.index
        return step, wasted_production_over_time

    def plot_wasted_production_over_time(self):
        return self._init_plot_wasted_production_over_time(init=False)

    def _init_plot_wasted_production_over_time(self, figsize=None, init=True):
        nom = str(inspect.currentframe().f_code.co_name).replace("_init_", "")
        fig, ax = self._get_fig(nom, figsize)
        step, wasted_production_over_time = self._get_data_wasted_production_over_time(
            init
        )

        line = ax.plot(
            step, wasted_production_over_time, label="wasted production", c="salmon"
        )
        if type(line) == list:
            line = line[0]
        max_ = wasted_production_over_time.max()
        self.figs[nom].append([line, max_])

        self._set_ax_format(ax)
        ax.set_xlim(-10, self.num_period * 1.1)
        ax.set_ylim(0, max_ * 1.1)
        ax.set_ylabel("Wasted Production (m³ of CO2)")
        ax.set_xlabel("Time (hours)")
        ax.set_title("Evolution of Wasted Production Over Time")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return nom
