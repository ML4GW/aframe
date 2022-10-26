from pathlib import Path

import h5py
from bokeh.layouts import column, row
from bokeh.models import Div, Panel, Select, Tabs
from vizapp.distributions import get_foreground, load_results
from vizapp.plots import BackgroundPlot, EventInspectorPlot, PerfSummaryPlot


class VizApp:
    def __init__(
        self,
        timeslides_dir: Path,
        data_dir: Path,
        sample_rate: float,
        fduration: float,
        valid_frac: float,
    ) -> None:
        self.timeslides_dir = timeslides_dir
        train_frac = 1 - valid_frac

        self.distributions = load_results(timeslides_dir)
        self.foregrounds = {}
        for norm, results in self.distributions.items():
            foreground = get_foreground(results, timeslides_dir, norm)
            self.foregrounds[norm] = foreground

        self.configure_widgets()
        self.configure_plots(sample_rate, fduration, train_frac, data_dir)
        self.update(None, None, self.norm_select.options[0])

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
            value=value,
            options=list(map(str, options)),
        )
        self.norm_select.on_change("value", self.update)
        self.widgets = row(header, self.norm_select)

    def configure_plots(self, sample_rate, fduration, train_frac, data_dir):
        self.perf_summary_plot = PerfSummaryPlot(300, 800)

        backgrounds = {}
        for ifo in ["H1", "L1"]:
            with h5py.File(data_dir / f"{ifo}_background.h5", "r") as f:
                bkgd = f["hoft"][:]
                bkgd = bkgd[: int(train_frac * len(bkgd))]
                backgrounds[ifo] = bkgd

        self.event_inspector = EventInspectorPlot(
            height=300,
            width=1500,
            data_dir=self.timeslides_dir,
            fduration=fduration,
            sample_rate=sample_rate,
            freq_low=30,
            freq_high=300,
            **backgrounds,
        )

        self.background_plot = BackgroundPlot(300, 1200, self.event_inspector)

        summary_tab = Panel(
            child=self.perf_summary_plot.layout, title="Summary"
        )

        analysis_layout = column(
            self.background_plot.layout, self.event_inspector.layout
        )
        analysis_tab = Panel(child=analysis_layout, title="Analysis")
        tabs = Tabs(tabs=[summary_tab, analysis_tab])
        self.layout = column(self.widgets, tabs)

    def update(self, attr, old, new):
        norm = None if new == "None" else float(new)
        foreground = self.foregrounds[norm]
        background = self.distributions[norm].background

        self.perf_summary_plot.update(foreground)
        self.background_plot.update(foreground, background, norm)
        self.event_inspector.reset()

    def __call__(self, doc):
        doc.add_root(self.layout)
