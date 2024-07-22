import io
from typing import TYPE_CHECKING, Sequence

import numpy as np
from bokeh.layouts import row
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure
from gwpy.plot import Plot
from PIL import Image
from plots.vizapp import palette

if TYPE_CHECKING:
    import gwpy
    from plots.vizapp.infer.analyzer import EventAnalyzer


class InspectorPlot:
    def __init__(self, analyzer: "EventAnalyzer"):
        self.analyzer = analyzer

    def initialize_sources(self):
        strain_source = {ifo: [] for ifo in self.analyzer.ifos}
        strain_source["t"] = []

        self.strain_source = ColumnDataSource(strain_source)
        self.response_source = ColumnDataSource(
            dict(nn=[], integrated=[], t=[])
        )
        self.spectrogram_source = ColumnDataSource(
            data=dict(image=[], x=[], y=[], dw=[], dh=[])
        )

    def get_layout(self, height: int, width: int) -> None:
        self.timeseries_plot = figure(
            title="Click on an event to inspect",
            height=height,
            width=width,
            y_range=(-5, 5),
            x_range=(-3, 5),
            x_axis_label="Time [s]",
            y_axis_label="Strain [unitless]",
        )
        self.timeseries_plot.toolbar.autohide = True

        items, self.strain_renderers = [], []
        for i, ifo in enumerate(self.analyzer.ifos):
            r = self.timeseries_plot.line(
                x="t",
                y=ifo,
                line_color=palette[i],
                line_alpha=0.6,
                legend_label=ifo,
                source=self.strain_source,
            )
            self.strain_renderers.append(r)
            items.append((ifo, [r]))

        self.timeseries_plot.extra_y_ranges = {"nn": Range1d(-1, 10)}
        self.timeseries_plot.add_layout(
            LinearAxis(axis_label="NN output", y_range_name="nn"), "right"
        )

        self.output_renderers = []
        for i, field in enumerate(["nn", "integrated"]):
            label = "NN output"
            if field == "integrated":
                label = "Integrated " + label

            r = self.timeseries_plot.line(
                "t",
                field,
                line_color=palette[2 + i],
                line_width=2,
                line_alpha=0.8,
                # legend_label=label,
                source=self.response_source,
                y_range_name="nn",
            )
            self.output_renderers.append(r)
            items.append((label, [r]))

        legend = Legend(items=items, orientation="horizontal")
        self.timeseries_plot.add_layout(legend, "below")

        hover = HoverTool(
            renderers=[r],
            tooltips=[
                ("NN response", "@nn"),
                ("Integrated NN response", "@integrated"),
            ],
        )
        self.timeseries_plot.add_tools(hover)
        self.timeseries_plot.legend.click_policy = "mute"

        self.spectrogram_plot = figure(
            height=height,
            width=width,
            title="Spectrogram",
            toolbar_location=None,
            x_axis_type=None,
            y_axis_type=None,
        )
        self.spectrogram_plot.grid.grid_line_color = None
        self.spectrogram_plot.outline_line_color = None
        self.spec_renderer = self.spectrogram_plot.image_rgba(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=self.spectrogram_source,
        )
        return row(self.timeseries_plot, self.spectrogram_plot)

    def plot(self, qscans: tuple["gwpy.spectrogram.Spectrogram"]):
        fig = Plot(
            *qscans,
            figsize=(10, 5),
            geometry=(1, len(self.analyzer.ifos)),
            yscale="log",
            method="pcolormesh",
            cmap="viridis",
        )
        for i, ax in enumerate(fig.axes):
            from matplotlib import ticker

            # TODO: account for half second somewhere
            ax.set_epoch(0.5)
            ax.set_title(self.analyzer.ifos[i])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, p: f"{x:.1f}")
            )

            if i == len(self.analyzer.ifos) - 1:
                fig.colorbar(ax=ax, label="Normalized energy")

        # save image png in memory so that
        # we can pass it to bokeh to plot
        out = io.BytesIO()
        fig.savefig(out, format="png", bbox_inches="tight")

        # open image, flip it right side up,
        # and format in a way that bokeh can plot
        image = Image.open(out)
        img_array = np.array(image)
        img_array = np.flipud(img_array)
        img_height, img_width, _ = img_array.shape
        img_array = img_array.view(dtype=np.uint32).reshape(
            (img_height, img_width)
        )
        return img_array

    def update(
        self,
        time: float,
        type: str,
        shifts: Sequence[float],
        title: str,
    ) -> None:
        foreground = type == "foreground"
        nn, integrated, whitened = self.analyzer.analyze(
            time, shifts, foreground
        )

        # update the strain source plot data
        # with the whitened strain, nn outputs,
        # and integrated outputs
        strain_source = {
            ifo: whitened[0][i][-len(self.analyzer.whitened_times) :]
            for i, ifo in enumerate(self.analyzer.ifos)
        }
        strain_source["t"] = self.analyzer.whitened_times

        # qscan whitened strain and plot spectrogram
        qscans = self.analyzer.qscan(strain_source)
        img = self.plot(qscans)

        width, height = img.shape[1], img.shape[0]
        self.spectrogram_plot.x_range = Range1d(0, width)
        self.spectrogram_plot.y_range = Range1d(0, height)
        self.spectrogram_plot.height = height
        self.spectrogram_plot.width = width

        spec_source = dict(image=[img], x=[0], y=[0], dw=[width], dh=[height])
        self.spectrogram_source.data = spec_source
        self.spec_renderer.data_source.data = spec_source

        self.strain_source.data = strain_source
        for r in self.strain_renderers:
            r.data_source.data = strain_source

        self.response_source.data = {
            "nn": nn,
            "integrated": integrated,
            "t": self.analyzer.inference_times,
        }
        for r in self.output_renderers:
            r.data_source.data = dict(
                nn=nn, integrated=integrated, t=self.analyzer.inference_times
            )

        # update axis labels, title and ranges of timeseries plot
        nn_min = nn.min()
        nn_max = nn.max()
        nn_min = 0.95 * nn_min if nn_min > 0 else 1.05 * nn_min

        nn_max = 1.05 * nn_max if nn_max > 0 else 0.95 * nn_max

        self.timeseries_plot.extra_y_ranges["nn"].start = nn_min
        self.timeseries_plot.extra_y_ranges["nn"].end = nn_max

        self.timeseries_plot.xaxis.axis_label = f"Time from {time:0.3f} [s]"

        self.timeseries_plot.title.text = title

    def reset(self):
        # TODO: implement this
        for r in self.strain_renderers:
            r.data_source.data = dict(H1=[], L1=[], t=[])

        for r in self.output_renderers:
            r.data_source.data = dict(nn=[], integrated=[], t=[])

        self.timeseries_plot.title.text = "Click on an event to inspect"
        self.timeseries_plot.xaxis.axis_label = "Time [s]"
