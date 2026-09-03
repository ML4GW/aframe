from pathlib import Path

from bilby.core.prior import PriorDict
from bokeh.io import save
from bokeh.layouts import gridplot
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet

from plots.core import style
from plots.core.constants import (
    SECONDS_PER_MONTH,
    SECONDS_PER_YEAR,
)
from plots.core.data import AnalysisData
from plots.core.gwtc3 import catalog_results
from plots.core.sv import compute_sensitive_volume


class SensitiveVolumePlot:
    def __init__(
        self,
        background: EventSet,
        foreground: RecoveredInjectionSet,
        rejected_params: InjectionParameterSet,
        mass_combos: list[tuple],
        source_prior: PriorDict,
        dt: float | None = None,
        # 100 per month, expressed in yr^-1
        max_far: float = 100 * SECONDS_PER_YEAR / SECONDS_PER_MONTH,
        sigma: float = 0.1,
    ):
        self.mass_combos = mass_combos
        self.source_prior = source_prior
        self.max_far = max_far
        self.sigma = sigma
        self.dt = dt

        self.data = AnalysisData(background, foreground, rejected_params)
        self.result = self.compute()
        self.grid = self.make_plot()

    @property
    def background(self):
        return self.data.background

    @property
    def foreground(self):
        return self.data.foreground

    @property
    def fars(self):
        return self.result.fars

    @property
    def thresholds(self):
        return self.result.thresholds

    @property
    def svs(self):
        return self.result.sv

    @property
    def errs(self):
        return self.result.err

    def compute(self):
        return compute_sensitive_volume(
            self.data,
            mass_combos=self.mass_combos,
            source_prior=self.source_prior,
            dt=self.dt,
            max_far=self.max_far,
            sigma=self.sigma,
        )

    def save(self, output_dir: Path):
        self.result.write(output_dir / "sensitive_volume.hdf5")

        # save grid plot as html
        save(self.grid, filename=output_dir / "sensitive_volume.html")

    def make_plot(self):
        plots = style.make_grid(self.mass_combos)
        for i, (p, color) in enumerate(
            zip(plots, style.palette, strict=False)
        ):
            fars = self.fars
            p.line(fars, self.svs[i], line_width=1.5, line_color=color)
            style.plot_err_bands(
                p,
                fars,
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
                    [fars[0], fars[-1]],
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

    def update(self, background, foreground):
        # TODO: this currently does nothing; need to rejigger the
        # figures to use ColumnDataSources so we can just push new data
        # into them on veto changes.
        pass
