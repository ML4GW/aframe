from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import h5py
import numpy as np
import torch
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
from gwpy.plot import Plot
from ledger.injections import InterferometerResponseSet, waveform_class_factory
from plots import palette

from .utils import get_indices, get_strain_fname

if TYPE_CHECKING:
    from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


class EventAnalyzer:
    def __init__(
        self,
        model: torch.nn.Module,
        whitener: "BatchWhitener",
        snapshotter: "BackgroundSnapshotter",
        strain_dir: Path,
        response_set: Path,
        psd_length: float,
        kernel_length: float,
        sample_rate: float,
        fduration: float,
        inference_sampling_rate: float,
        integration_length: float,
        batch_size: int,
        device: str,
        ifos: List[str],
        padding: int = 3,
    ):
        self.model = model
        self.whitener = whitener
        self.snapshotter = snapshotter
        self.strain_dir = strain_dir
        self.response_set = response_set
        self.ifos = ifos

        self.padding = padding
        self.sample_rate = sample_rate
        self.fduration = fduration
        self.psd_length = psd_length
        self.kernel_length = kernel_length
        self.inference_sampling_rate = inference_sampling_rate
        self.integration_length = integration_length
        self.batch_size = batch_size
        self.device = device

    @property
    def waveform_class(self):
        return waveform_class_factory(
            self.ifos, InterferometerResponseSet, "IfoWaveformSet"
        )

    @property
    def kernel_size(self):
        return int(self.kernel_length * self.sample_rate)

    @property
    def state_shape(self):
        return (1, len(self.ifos), self.snapshotter.state_size)

    @property
    def inference_stride(self):
        return int(self.sample_rate / self.inference_sampling_rate)

    @property
    def step_size(self):
        return int(self.batch_size * self.inference_stride)

    @property
    def integration_size(self):
        return int(self.integration_length * self.inference_sampling_rate)

    @property
    def window(self):
        return np.ones((self.integration_size,)) / self.integration_size

    @property
    def times(self):
        """
        Returns the time values relative to event time
        """
        start = (
            self.psd_length
            + self.kernel_length
            + (self.fduration / 2)
            + self.padding
        )
        stop = self.kernel_length + (self.fduration / 2) + self.padding
        return np.arange(-start, stop, 1 / self.sample_rate)

    @property
    def inference_times(self):
        return self.times[:: self.inference_stride]

    def find_strain(self, time: float, shifts: Sequence[float]):
        # find strain file corresponding to requested time
        fname, t0, duration = get_strain_fname(self.strain_dir, time)
        # find indices of data needed for inference
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)
        start, stop = get_indices(
            times, time + self.times[0], time + self.times[-1]
        )
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

        return torch.stack(strain, axis=0), time + self.times[0]

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
        return integrated[: -self.integration_size + 1]

    def infer(self, X: torch.Tensor):
        ys, batches = [], []
        start = 0
        state = torch.zeros(self.state_shape).to(self.device)
        # pad X up to batch size
        remainder = X.shape[-1] % self.step_size
        num_slice = None
        if remainder:
            pad = self.step_size - remainder
            X = torch.nn.functional.pad(X, (0, pad))
            num_slice = pad // self.inference_stride
        slc = slice(-num_slice)

        while start <= (X.shape[-1] - self.step_size):
            stop = start + self.step_size
            x = X[:, :, start:stop]
            with torch.no_grad():
                x, state = self.snapshotter(x, state)
                batch = self.whitener(x)
                y_hat = self.model(batch)[:, 0].cpu().numpy()

            batches.append(batch.cpu().numpy())
            ys.append(y_hat)
            start += self.step_size

        batches = np.concatenate(batches)[slc]
        ys = np.concatenate(ys)[slc]
        return ys, batches

    def analyze(self, time, shifts, foreground):
        strain, t0 = self.find_strain(time, shifts)
        if foreground:
            waveform = self.find_waveform(time, shifts)
            strain = waveform.inject(strain, t0)
        strain = strain[None]
        strain = torch.Tensor(strain).to(self.device)
        outputs, whitened = self.infer(strain)
        integrated = self.integrate(outputs)

        return outputs, whitened, integrated


class InspectorPlot:
    def __init__(self, page, analyzer: EventAnalyzer):
        self.page = page
        self.analyzer = analyzer

    def initialize_sources(self):
        strain_source = {ifo: [] for ifo in self.analyzer.ifos}
        strain_source["t"] = []

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
        nn, whitened, integrated = self.analyzer.analyze(
            event_time, shift, foreground
        )
        # det = integrated.max()
        # self.plot(qscans, det)
        # self.spectrogram.text = f"<img src=myapp/{qscan_path}>"

        # TODO: best way to show whitened data?
        """
        strain_source = {
            ifo: whitened[i] for
            i, ifo in enumerate(self.analyzer.ifos)
        }
        strain_source["t"] = self.analyzer.times
        self.strain_source.data = strain_source
        for r in self.strain_renderers:
            r.data_source.data = strain_source
        """

        self.response_source.data = {
            "nn": nn,
            "integrated": integrated,
            "t": self.analyzer.inference_times,
        }
        for r in self.output_renderers:
            r.data_source.data = dict(
                nn=nn, integrated=integrated, t=self.analyzer.inference_times
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
