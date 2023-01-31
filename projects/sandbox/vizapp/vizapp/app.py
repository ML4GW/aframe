import copy
import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import bilby
import h5py
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Div, MultiChoice, Panel, Select, Tabs
from vizapp.distributions import get_foreground, load_results
from vizapp.plots import (
    BackgroundPlot,
    EventInspectorPlot,
    PerfSummaryPlot,
    VolumeTimeVsFAR,
)

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology
    from vizapp.vetoes import VetoParser


class VizApp:
    def __init__(
        self,
        cosmology: "Cosmology",
        source_prior: "bilby.core.prior.PriorDict",
        timeslides_results_dir: Path,
        timeslides_strain_dir: Path,
        train_data_dir: Path,
        ifos: List[str],
        sample_rate: float,
        fduration: float,
        valid_frac: float,
        veto_parser: "VetoParser",
    ) -> None:
        self.logger = logging.getLogger("vizapp")
        self.logger.debug("Loading analyzed distributions")
        self.veto_parser = veto_parser
        self.ifos = ifos
        self.source_prior = source_prior

        # load in foreground and background distributions
        self.distributions = load_results(timeslides_results_dir)

        self.logger.debug("Structuring distribution events")
        self.foregrounds = {}
        for norm, results in self.distributions.items():

            foreground = get_foreground(
                results, timeslides_strain_dir, timeslides_results_dir, norm
            )
            self.foregrounds[norm] = foreground

        self.logger.debug("Configuring widgets")
        self.configure_widgets()

        self.logger.debug("Calculating all veto combinations")
        self.calculate_veto_distributions()

        self.logger.debug("Configuring plots")
        self.configure_plots(
            sample_rate,
            fduration,
            1 - valid_frac,
            train_data_dir,
            timeslides_strain_dir,
            timeslides_results_dir,
        )
        self.update_norm(None, None, self.norm_select.options[0])

        self.logger.info("Application ready!")

    def configure_widgets(self):
        header = Div(text="<h1>BBHNet Performance Dashboard</h1>", width=500)

        norm_options = list(self.distributions)
        if None in norm_options:
            value = None
            options = [None] + sorted([i for i in norm_options if i])
        else:
            options = sorted(norm_options)
            value = options[0]

        self.norm_select = Select(
            title="Normalization period [s]",
            value=str(value),
            options=list(map(str, options)),
        )
        self.norm_select.on_change("value", self.update_norm)

        self.veto_labels = ["CAT1"]  # ["CAT1", "CAT2", "CAT3", "GATES"]
        self.veto_choices = MultiChoice(
            title="Applied Vetoes", value=[], options=self.veto_labels
        )
        self.veto_choices.on_change("value", self.update_vetoes)
        self.widgets = row(header, self.norm_select, self.veto_choices)

    # Calculate all combinations of vetoes for each norm up front
    # so changing vetoe configurations in app is faster

    # TODO: This could also probably be a part of the
    # analysis project, and just loaded in here.
    def calculate_veto_distributions(self):

        self.vetoed_distributions = {}
        self.vetoed_foregrounds = {}

        # create all combos of vetoes
        for n in range(len(self.veto_labels) + 1):
            combos = list(itertools.combinations(self.veto_labels, n))
            for combo in combos:
                # sort vetoes and join to create label
                veto_label = "_".join(sorted(combo))
                self.logger.debug(
                    f"Calculating vetoe comboe {veto_label} for all norms"
                )
                # create vetoed foreground and background distributions
                self.vetoed_distributions[veto_label] = {}
                self.vetoed_foregrounds[veto_label] = {}
                # calculate this vetoe combo for each norm and store
                for norm, result in self.distributions.items():

                    background = copy.deepcopy(result.background)
                    for category in combo:

                        vetoes = self.veto_parser.get_vetoes(category)
                        background.apply_vetoes(**vetoes)

                    foreground = copy.deepcopy(self.foregrounds[norm])

                    foreground.fars = background.far(
                        foreground.detection_statistics
                    )
                    self.vetoed_foregrounds[veto_label][norm] = foreground
                    self.vetoed_distributions[veto_label][norm] = background

    def configure_plots(
        self,
        sample_rate,
        fduration,
        train_frac,
        train_data_dir,
        timeslides_strain_dir,
        timeslides_results_dir,
    ):
        self.perf_summary_plot = PerfSummaryPlot(300, 800)
        self.volume_time_vs_far = VolumeTimeVsFAR(
            300, 800, source_prior=self.source_prior, cosmology=self.cosmology
        )

        backgrounds = {}
        for ifo in self.ifos:
            with h5py.File(train_data_dir / f"{ifo}_background.h5", "r") as f:
                bkgd = f["hoft"][:]
                bkgd = bkgd[: int(train_frac * len(bkgd))]
                backgrounds[ifo] = bkgd

        self.event_inspector = EventInspectorPlot(
            height=300,
            width=1500,
            response_dir=timeslides_results_dir,
            strain_dir=timeslides_strain_dir,
            fduration=fduration,
            sample_rate=sample_rate,
            freq_low=30,
            freq_high=300,
            **backgrounds,
        )

        self.background_plot = BackgroundPlot(300, 1200, self.event_inspector)

        summary_layout = column(
            [self.perf_summary_plot.layout, self.volume_time_vs_far.layout]
        )
        summary_tab = Panel(child=summary_layout, title="Summary")

        analysis_layout = column(
            self.background_plot.layout, self.event_inspector.layout
        )
        analysis_tab = Panel(child=analysis_layout, title="Analysis")
        tabs = Tabs(tabs=[summary_tab, analysis_tab])
        self.layout = column(self.widgets, tabs)

    def update_norm(self, attr, old, new):
        current_veto_label = "_".join(sorted(self.veto_choices.value))
        norm = None if new == "None" else float(new)

        self.logger.debug(f"Updating plots with normalization value {norm}")
        background = self.vetoed_distributions[current_veto_label][norm]
        foreground = self.vetoed_foregrounds[current_veto_label][norm]

        self.perf_summary_plot.update(foreground)
        self.volume_time_vs_far.update(foreground)
        self.background_plot.update(foreground, background, norm)
        self.event_inspector.reset()

    def update_vetoes(self, attr, old, new):

        # retrieve the current normalization value
        current_norm = float(self.norm_select.value)

        # calculate vetoe label for this combo
        veto_label = "_".join(sorted(new))
        self.logger.debug(f"Applying veto comboe {veto_label}")

        # get background and foreground for this veto label
        background = self.vetoed_distributions[veto_label][current_norm]
        foreground = self.vetoed_foregrounds[veto_label][current_norm]
        self.logger.debug(f"{np.mean(foreground.fars)}")
        # update plots
        self.logger.debug(
            "Updating plots with new distributions after changing vetoes"
        )
        # update plots
        self.perf_summary_plot.update(foreground)
        self.volume_time_vs_far.update(foreground)
        self.background_plot.update(foreground, background, current_norm)
        self.event_inspector.reset()

    def __call__(self, doc):
        doc.add_root(self.layout)
