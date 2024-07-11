from bokeh.layouts import column
from plots.pages.analysis.distribution import DistributionPlot
from plots.pages.analysis.inspector import EventAnalyzer, InspectorPlot
from plots.pages.page import Page


class Analysis(Page):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        analyzer = self.get_analyzer()
        self.event_inspector = InspectorPlot(self, analyzer)
        self.distribution_plot = DistributionPlot(self, self.event_inspector)

        self.initialize_sources()

    def get_analyzer(self):
        return EventAnalyzer(
            self.app.model,
            self.app.whitener,
            self.app.snapshotter,
            self.app.data.data_dir / "background",
            self.app.data.response_set,
            self.app.data.psd_length,
            self.app.data.kernel_length,
            self.app.data.sample_rate,
            self.app.data.fduration,
            self.app.data.inference_sampling_rate,
            self.app.data.integration_length,
            self.app.data.batch_size,
            self.app.data.device,
            self.app.data.ifos,
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
        return column(distribution, event_inspector)

    def update(self):
        self.distribution_plot.update()
