from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import bilby
    from astropy.cosmology import Cosmology

import logging

from bilby.gw.conversion import luminosity_distance_to_redshift
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, HoverTool, NumericInput
from bokeh.plotting import figure

from bbhnet.analysis.sensitivity import SensitiveVolumeCalculator
from bbhnet.priors.priors import gaussian_masses

MPC3_TO_GPC3 = 1e-9


class VolumeTimeVsFAR:
    def __init__(
        self,
        height,
        width,
        source_prior: "bilby.core.prior.PriorDict",
        cosmology: "Cosmology",
    ):
        self.height = height
        self.width = width
        self.source_prior = source_prior
        self.cosmology = cosmology
        self.keys = self.source_prior.keys()

        self.fars = np.logspace(0, 7, 7)
        self.configure_sources()
        self.configure_widgets()
        self.configure_plots()
        self.logger = logging.getLogger("V vs FAR")
        self.logger.debug(self.fars)

    def configure_plots(self):
        self.figure = figure(
            height=self.height,
            width=self.width,
            x_axis_type="log",
            y_axis_type="log",
            x_axis_label="FAR (yr ^-1)",
            y_axis_label="<V> Gpc^3",
        )

        self.figure.line(
            x="far",
            y="vt",
            source=self.source,
            line_width=2,
            line_alpha=0.6,
            line_color="blue",
        )

        # Add error bars

        self.figure.segment(
            source=self.source,
            x0="far",
            y0="error_low",
            x1="far",
            y1="error_high",
            line_width=2,
        )

        self.layout = row(self.widgets, self.figure)

    def configure_widgets(self):
        self.m1_selector = NumericInput(
            title="Enter mass_1", value=30, low=5, high=100
        )
        self.m2_selector = NumericInput(
            title="Enter mass_2 (must be below mass_1)",
            value=30,
            low=5,
            high=100,
        )

        self.sd_selector = NumericInput(
            title="Enter standard deviation",
            value=2,
            low=1,
            high=10,
        )

        self.calculate_button = Button(label="Calculate VT")
        self.calculate_button.on_click(self.calculate_vt)
        self.widgets = column(
            self.m1_selector,
            self.m2_selector,
            self.sd_selector,
            self.calculate_button,
        )

    def configure_sources(self):
        self.source = ColumnDataSource(
            data=dict(
                far=[],
                volume=[],
                error_low=[],
                error_high=[],
                uncertainty=[],
            )
        )

    def _to_redshift(self, x):
        return luminosity_distance_to_redshift(x, self.cosmology)

    def calculate_vt(self, event):
        m1_mean = self.m1_selector.value
        m2_mean = self.m2_selector.value
        sigma = self.sd_selector.value

        # TODO: handle this better
        if m1_mean < m2_mean:
            self.logger.error("m1 must be greater than m2")
            return

        self.logger.debug(
            f"Calculating VT for m1 = {m1_mean}, m2 = {m2_mean}, sd = {sigma}"
        )
        target = gaussian_masses(m1_mean, m2_mean, sigma, self.cosmology)

        fars = []
        volumes = []
        uncertainties = []
        n_effs = []

        self.logger.debug("Converting distances to redshifts")
        redshifts = self._to_redshift(self.foreground.distances)
        for far in self.fars:
            self.figure.title.text = (
                f"VT vs FAR for m1 = {m1_mean}, m2 = {m2_mean},  sd = {sigma}"
            )
            # downselect to injections that are detected at this FAR
            indices = self.foreground.fars < far

            # parse foreground statistics into a dictionary
            # compatible with bilbys prior.prob method
            recovered_parameters = {
                "mass_1": self.foreground.m1s[indices],
                "mass_2": self.foreground.m2s[indices],
                "redshift": redshifts[indices],
            }

            logging.debug(f"Computing V for FAR {far}")
            sensitive_volume_calc = SensitiveVolumeCalculator(
                source=self.source_prior,
                recovered_parameters=recovered_parameters,
                n_injections=self.n_injections,
                cosmology=self.cosmology,
            )

            (
                volume,
                uncertainty,
                n_eff,
            ) = sensitive_volume_calc.calculate_sensitive_volume(target=target)

            # convert volume into Gpc^3
            volume *= MPC3_TO_GPC3
            uncertainty *= MPC3_TO_GPC3
            fars.append(far)
            volumes.append(volume)
            uncertainties.append(uncertainty)
            n_effs.append(n_eff)

        volumes = np.array(volumes)
        uncertainties = np.array(uncertainties)
        n_eff = np.array(n_effs)
        self.logger.debug(uncertainties)
        self.source.data = {
            "far": fars,
            "volume": volumes,
            "error_low": volumes - uncertainties,
            "error_high": volumes + uncertainties,
            "n_eff": n_effs,
            "uncertainty": uncertainties,
        }

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Volume", "@volume"),
                ("N_eff", "@n_eff"),
                ("Uncertainty", "@uncertainty"),
            ]
        )
        self.figure.add_tools(hover)

    def update(self, foreground):
        self.foreground = foreground
        self.n_injections = len(foreground.injection_times)
        self.livetime = foreground.livetime
        self.calculate_vt(None)
