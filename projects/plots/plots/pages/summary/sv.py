import logging
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
from astropy.cosmology import Planck15 as cosmology
from bilby.core.prior import PriorDict
from bokeh.io import save
from bokeh.layouts import gridplot
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from priors.priors import log_normal_masses

from . import compute, utils
from .gwtc3 import catalog_results


def get_prob(prior, ledger):
    sample = dict(mass_1=ledger.mass_1, mass_2=ledger.mass_2)
    return prior.prob(sample, axis=0)


class SensitiveVolumePlot:
    def __init__(
        self,
        background: EventSet,
        foreground: RecoveredInjectionSet,
        rejected_params: InjectionParameterSet,
        mass_combos: List[tuple],
        source_prior: PriorDict,
        dt: Optional[float] = None,
        max_far: float = 1e-6,
        sigma: float = 0.1,
    ):
        self.background = background
        self.foreground = foreground
        self.rejected_params = rejected_params
        self.mass_combos = mass_combos
        self.source_prior = source_prior
        self.max_far = max_far
        self.sigma = sigma
        self.dt = dt

        self._thresholds = None

        self.source_probs = get_prob(source_prior, foreground)
        self.source_rejected_probs = get_prob(
            self.source_prior, rejected_params
        )

        self.weights = np.zeros((len(mass_combos), len(self.source_probs)))
        self.v0 = self.calc_v0()
        self.svs, self.errs = self.compute_sv()
        self.grid = self.make_plot()

    @property
    def thresholds(self):
        if self._thresholds is not None:
            return self._thresholds
        self._thresholds = np.sort(self.background.detection_statistic)[
            -self.max_events :
        ][::-1]
        return self._thresholds

    @property
    def Tb(self):
        return self.background.Tb

    @property
    def max_events(self):
        return int(self.max_far * self.background.Tb)

    @property
    def fars(self):
        return np.arange(1, self.max_events + 1) / self.Tb

    @property
    def zprior(self):
        return self.source_prior["redshift"]

    def calc_v0(self):
        zprior = self.source_prior["redshift"]
        zmin, zmax = zprior.minimum, zprior.maximum
        try:
            decprior = self.source_prior["dec"]
        except KeyError:
            decrange = None
        else:
            decrange = (decprior.minimum, decprior.maximum)
        v0 = utils.get_astrophysical_volume(zmin, zmax, cosmology, decrange)
        v0 /= 10**9
        return v0

    def compute_sv(self):
        for i, combo in enumerate(self.mass_combos):
            logging.info(f"Computing likelihoods under {combo} log normal")
            prior, _ = log_normal_masses(
                *combo, sigma=self.sigma, cosmology=cosmology
            )
            prob = get_prob(prior, self.foreground)
            rejected_prob = get_prob(prior, self.rejected_params)

            weight = prob / self.source_probs

            rejected_weights = rejected_prob / self.source_rejected_probs
            norm = weight.sum() + rejected_weights.sum()
            weight /= norm

            # finally, enforce recovery time delta by setting weights to 0
            # for events outside of the delta t
            if self.dt is not None:
                logging.info(
                    f"Enforcing recovery time delta of {self.dt} seconds"
                )
                mask = (
                    np.abs(
                        self.foreground.detection_time
                        - self.foreground.injection_time
                    )
                    <= self.dt
                )
                weight[~mask] = 0

            self.weights[i] = weight

        logging.info("Computing sensitive volume at thresholds")
        y, err = compute.sensitive_volume(
            self.foreground.detection_statistic, self.weights, self.thresholds
        )
        y *= self.v0
        err *= self.v0
        return y, err

    def save(self, output_dir: Path):
        # save raw data to h5
        with h5py.File(output_dir / "sensitive_volume.h5", "w") as f:
            f.create_dataset("thresholds", data=self.thresholds)
            f.create_dataset("fars", data=self.fars)
            for i, combo in enumerate(self.mass_combos):
                g = f.create_group("-".join(map(str, combo)))
                g.create_dataset("sv", data=self.y[i])
                g.create_dataset("err", data=self.err[i])

        # save grid plot as html
        save(self.grid, filename=output_dir / "sensitive_volume.html")

    def make_plot(self):
        plots = utils.make_grid(self.mass_combos)
        for i, (p, color) in enumerate(zip(plots, utils.palette)):
            p.line(self.fars, self.svs[i], line_width=1.5, line_color=color)
            utils.plot_err_bands(
                p,
                self.fars,
                self.svs[i],
                self.errs[i],
                line_color=color,
                line_width=0.8,
                fill_color=color,
                fill_alpha=0.4,
            )

            for pipeline, data in catalog_results.items():
                # convert VT to volume by dividing out years
                vt = data["vt"][self.mass_combos[i]]
                v = vt * 365 / data["Tb"]

                # only include a legend on the top left
                kwargs = {}
                if i == 0:
                    kwargs["legend_label"] = pipeline
                p.line(
                    [self.fars[0], self.fars[-1]],
                    [v, v],
                    line_color="#333333",
                    line_dash=data["dash"],
                    line_alpha=0.7,
                    line_width=2,
                    **kwargs,
                )

                # style the legend on the top left plot
                if i == 0:
                    # style legend position
                    p.legend.location = "top_left"
                    p.legend.margin = 4
                    p.legend.padding = 2

                    # style individual glyphs
                    p.legend.glyph_height = 6
                    p.legend.label_text_font_size = "8pt"
                    p.legend.label_height = 8

                    # style title
                    p.legend.title = "GWTC-3 comparisons"
                    p.legend.title_text_font_size = "9pt"
                    p.legend.title_text_font_style = "bold"

        grid = gridplot(plots, toolbar_location="right", ncols=2)
        return grid

    def get_layout(self):
        return self.grid

    def update(self):
        pass