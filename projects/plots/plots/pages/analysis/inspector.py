from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import h5py
import numpy as np
import torch
from ledger.injections import waveform_class_factory
from plots.data import Data

from .utils import get_strain_fname, get_indices
from gwpy.plot import Plot
from bokeh.layouts import row
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    Legend,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure
from plots import palette

if TYPE_CHECKING:
    from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


class EventAnalyzer:
    def __init__(
        self,
        model: torch.nn.Module,
        whitener: BatchWhitener,
        snapshotter: BackgroundSnapshotter,
        strain_dir: Path,
        response_set: Path,
        sample_rate: float,
        fduration: float,
        ifos: List[str],
    ):
        
        self.model = model
        self.whitener = whitener
        self.snapshotter = snapshotter
        self.strain_dir = strain_dir
        self.response_set = response_set
        self.ifos = ifos

        self.sample_rate = sample_rate
        self.fduration = fduration

    @property
    def waveform_class(self):
        return waveform_class_factory(self.ifos)

    def find_strain(self, time: float, shifts: Sequence[float]):
        # find strain file corresponding to requested time
        fname, t0, duration = get_strain_fname(self.strain_dir, time)
        start, stop = None
        # find indices of data needed for inference
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)
        start, stop = get_indices(
            times,
            time - self.length_previous_data,
            time + self.padding + (self.fduration / 2),
        )
        times = times[start:stop]
        strain = []
        with h5py.File(fname, "r") as f:
            for ifo, shift in zip(self.ifos, shifts):
                shift_size = int(shift * self.sample_rate)
                start_shifted, stop_shifted = (
                    start + shift_size,
                    stop + shift_size,
                )
                data = torch.tensor(f[ifo][start_shifted:stop_shifted])
                strain.append(data)

        return torch.stack(strain, axis=0), times[0]

    def find_waveform(self, time: float, shifts: np.ndarray):
        """
        find the closest injection that corresponds to event
        time and shifts from waveform dataset
        """
        waveform = self.waveform_class.read(
            self.response_set, time - 0.1, time + 0.1, shifts
        )
        return waveform
    
    def integrate(self, y):
        integrated = np.convolve(y, self.window, mode="full")
        return integrated[: -self.window_size + 1]
        
    def infer(self, X: torch.Tensor):
        ys, batches = [], []
        start = 0
        state = torch.zeros(self.state_shape)
        while start < (X.shape[-1] - self.step_size):
            stop = start + self.step_size
            x = X[:, :, start:stop]
            with torch.no_grad():
                x = torch.Tensor(x)
                x, state = self.snapshotter(x, state)
                batch = self.preprocessor(x)
                y_hat = self.model(batch)[:, 0].numpy()

            batches.append(batch.numpy())
            ys.append(y_hat)
            start += self.step_size
        batches = np.concatenate(batches)
        ys = np.concatenate(ys)
        return ys, batches

    def analyze(self, time, shifts, foreground):
        strain, t0 = self.find_strain(time, shifts)
        if foreground:
            waveform = self.find_waveform(time, shifts)
            strain = waveform.inject(strain, t0)
        strain = strain[None]

        outputs, whitened = self.infer(strain)
        integrated = self.integrate(outputs)
        outputs = outputs[self.integration_size :]

        return outputs, whitened, integrated


class InspectorPlot:
    def __init__(self, page, analyzer: EventAnalyzer):
        self.page = page
        self.analyzer = analyzer

    def initialize_sources(self):
        strain_source = {ifo: [] for ifo in self.analyzer.ifos}.update({"t": []})
        self.strain_source = ColumnDataSource(strain_source)
        self.response_source = ColumnDataSource(
            dict(nn=[], integrated=[], t=[])
        )
        self.spectrogram = Div(text="", width=400, height=400)

    def get_layout(self, height: int, width: int) -> None:
        self.timeseries_plot = figure(
            title="Click on an event to inspect",
            height=height,
            width=width,
            y_range=(-5, 5),
            x_range=(-3, 3),
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

        return row(self.timeseries_plot, self.spectrogram)

    def plot(self, qscans, det):
        fig = Plot(
            *qscans,
            figsize=(12, 6),
            geometry=(1, 2),
            yscale="log",
            method="pcolormesh",
            cmap="viridis",
        )
        fig.savefig(self.analyzer.qscan_dir / f"qscan-{det}.png")

    def update(
        self,
        event_time: float,
        event_type: str,
        shift: np.ndarray,
        title: str,
    ) -> None:
        foreground = event_type == "foreground"
        (
            nn,
            integrated,
            whitened,
            times,
            inference_times,
            # qscans,
        ) = self.analyzer(event_time, shift, foreground)
        # det = integrated.max()
        # self.plot(qscans, det)

        # self.spectrogram.text = f"<img src=myapp/{qscan_path}>"

        h1, l1 = whitened
        # normalize times with respect to event time
        times = times - event_time
        inference_times = inference_times - event_time

        self.strain_source.data = {"H1": h1, "L1": l1, "t": times}
        for r in self.strain_renderers:
            r.data_source.data = {"H1": h1, "L1": l1, "t": times}

        self.response_source.data = {
            "nn": nn,
            "integrated": integrated,
            "t": inference_times,
        }
        for r in self.output_renderers:
            r.data_source.data = dict(
                nn=nn, integrated=integrated, t=inference_times
            )

        nn_min = nn.min()
        nn_max = nn.max()
        nn_min = 0.95 * nn_min if nn_min > 0 else 1.05 * nn_min

        nn_max = 1.05 * nn_max if nn_max > 0 else 0.95 * nn_max

        self.timeseries_plot.extra_y_ranges["nn"].start = nn_min
        self.timeseries_plot.extra_y_ranges["nn"].end = nn_max

        self.timeseries_plot.xaxis.axis_label = (
            f"Time from {event_time:0.3f} [s]"
        )

        self.timeseries_plot.title.text = title

    def reset(self):
        # TODO: implement this
        for r in self.strain_renderers:
            r.data_source.data = dict(H1=[], L1=[], t=[])

        for r in self.output_renderers:
            r.data_source.data = dict(nn=[], integrated=[], t=[])

        self.timeseries_plot.title.text = "Click on an event to inspect"
        self.timeseries_plot.xaxis.axis_label = "Time [s]"
