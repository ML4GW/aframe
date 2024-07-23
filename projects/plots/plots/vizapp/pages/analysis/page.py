from typing import TYPE_CHECKING

from bokeh.layouts import column
from plots.vizapp.infer.analyzer import EventAnalyzer
from plots.vizapp.pages.analysis.distribution import DistributionPlot
from plots.vizapp.pages.analysis.inspector import InspectorPlot
from plots.vizapp.pages.page import Page

if TYPE_CHECKING:
    from ledger.events import EventSet, RecoveredInjectionSet


class Analysis(Page):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        analyzer = self.get_analyzer()
        self.event_inspector = InspectorPlot(analyzer)
        self.distribution_plot = DistributionPlot(self, self.event_inspector)
        self.initialize_sources()

    def get_analyzer(self):
        return EventAnalyzer(
            self.app.model,
            self.app.data_dir / "background",
            self.app.data_manager.response_set,
            self.app.psd_length,
            self.app.kernel_length,
            self.app.sample_rate,
            self.app.fduration,
            self.app.inference_sampling_rate,
            self.app.integration_length,
            self.app.batch_size,
            self.app.highpass,
            self.app.fftlength,
            self.app.device,
            self.app.ifos,
        )

    def initialize_sources(self) -> None:
        self.distribution_plot.initialize_sources()
        self.event_inspector.initialize_sources()

    def get_layout(self):
        event_inspector = self.event_inspector.get_layout(
            height=400, width=600
        )
        distribution = self.distribution_plot.get_layout(
            height=400, width=1500
        )
        return column(
            distribution, event_inspector, sizing_mode="stretch_both"
        )

    def update(
        self, background: "EventSet", foreground: "RecoveredInjectionSet"
    ):
        self.distribution_plot.update(background, foreground)
