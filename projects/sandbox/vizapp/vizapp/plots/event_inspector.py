from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, sosfiltfilt, windows
from vizapp import palette, path_utils


class Preprocessor:
    def __init__(
        self,
        sample_rate: float,
        fduration: float = 1,
        freq_low: Optional[float] = None,
        freq_high: Optional[float] = None,
        **backgrounds: np.ndarray,
    ) -> None:
        self.sample_rate = sample_rate
        self.window = windows.hann(int(2 * fduration * sample_rate))

        self.backgrounds = {}
        for ifo, background in backgrounds.items():
            background = TimeSeries(background, sample_rate=sample_rate)
            background = background.asd(
                fftlength=2, method="median", window="hann"
            )
            self.backgrounds[ifo] = background

        if freq_low is None and freq_high is None:
            self.sos = None
        else:
            if freq_low is not None and freq_high is not None:
                freq = (freq_low, freq_high)
                btype = "bandpass"
            elif freq_low is not None:
                freq = freq_low
                btype = "highpass"
            elif freq_high is not None:
                freq = freq_high
                btype = "lowpass"

            self.sos = butter(
                8, freq, fs=sample_rate, btype=btype, output="sos"
            )

    def __call__(self, **strains):
        ys = []
        pad = int(len(self.window) // 2)
        for ifo, strain in strains.items():
            strain = TimeSeries(strain, sample_rate=self.sample_rate)
            strain = strain.whiten(asd=self.backgrounds[ifo]).value

            strain[:pad] *= self.window[:pad]
            strain[-pad:] *= self.window[-pad:]

            strain = sosfiltfilt(self.sos, strain)
            strain = strain[pad:-pad]
            ys.append(strain)

        return tuple(ys)


def get_indices(t, lower, upper):
    mask = (lower <= t) & (t < upper)
    idx = np.where(mask)[0]
    return idx[0], idx[-1]


class EventInspectorPlot:
    def __init__(
        self,
        height: int,
        width: int,
        data_dir: Path,
        sample_rate: float,
        fduration: float = 1,
        freq_low: Optional[float] = None,
        freq_high: Optional[float] = None,
        **backgrounds: np.ndarray,
    ) -> None:
        self.data_dir = data_dir
        self.fduration = fduration
        self.preprocessor = Preprocessor(
            sample_rate, fduration, freq_low, freq_high, **backgrounds
        )

        self.configure_sources()
        self.configure_plots(height, width)

    def configure_sources(self):

        self.strain_source = ColumnDataSource(dict(H1=[], L1=[], t=[]))
        self.response_source = ColumnDataSource(
            dict(nn=[], integrated=[], t=[])
        )

    def configure_plots(self, height: int, width: int) -> None:

        self.timeseries_plot = figure(
            title="Click on an event to inspect",
            height=height,
            width=int(width / 3),
            y_range=(-2, 2),
            x_range=(-3, 3),
            x_axis_label="Time [s]",
            y_axis_label="Strain [unitless]",
            tools="",
        )
        self.timeseries_plot.toolbar.autohide = True

        items, self.strain_renderers = [], []
        for i, ifo in enumerate(["H1", "L1"]):
            r = self.timeseries_plot.line(
                x="t",
                y=ifo,
                line_color=palette[i],
                line_alpha=0.6,
                # legend_label=ifo,
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

        self.layout = self.timeseries_plot

    def load_nn_response(
        self, fname: Path, event_time: float, pad: float = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(fname, "r") as f:
            t = f["GPSstart"][:]
            start, stop = get_indices(t, event_time - pad, event_time + pad)

            nn = f["out"][start:stop]
            y = f["integrated"][start:stop]
            t = t[start:stop]
        return y, nn, t

    def get_strain(
        self, event_time: float, event_type: str, shift: str, pad: float = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        strain_dirname = path_utils.get_strain_dirname(event_type)
        strain_dir = self.data_dir / shift / strain_dirname
        strain_fname = path_utils.get_event_fname(strain_dir, event_time)

        pad = pad + self.fduration
        with h5py.File(strain_fname, "r") as f:
            # decrease our read time by figuring out which
            # indices from the strain arrays we need to slice
            # beforehand
            t = f["GPSstart"][:]
            start, stop = get_indices(t, event_time - pad, event_time + pad)

            # now load just the slices we'll need
            h1 = f["H1"][start:stop]
            l1 = f["L1"][start:stop]
            t = t[start:stop]

        h1, l1 = self.preprocessor(H1=h1, L1=l1)
        drop = int(len(t) - len(h1)) // 2
        t = t[drop:-drop]
        return h1, l1, t

    def get_nn_response(
        self,
        event_time: float,
        event_type: str,
        shift: str,
        norm: Optional[float] = None,
        pad: float = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        response_dirname = path_utils.get_response_dirname(event_type, norm)
        response_dir = self.data_dir / shift / response_dirname
        response_fname = path_utils.get_event_fname(response_dir, event_time)
        with h5py.File(response_fname, "r") as f:
            t = f["GPSstart"][:]
            start, stop = get_indices(t, event_time - pad, event_time + pad)

            nn = f["out"][start:stop]
            y = f["integrated"][start:stop]
            t = t[start:stop]
        return y, nn, t

    def update(
        self,
        event_time: float,
        event_type: str,
        shift: np.ndarray,
        norm: Optional[float] = None,
        **metadata,
    ) -> None:
        if not isinstance(shift, str):
            shift = "dt-H{}-L{}".format(*shift)

        h1, l1, t = self.get_strain(event_time, event_type, shift)
        t = t - event_time
        self.strain_source.data = {"H1": h1, "L1": l1, "t": t}

        for r in self.strain_renderers:
            r.data_source.data = {"H1": h1, "L1": l1, "t": t}

        y, nn, t = self.get_nn_response(event_time, event_type, shift, norm)
        t = t - event_time
        self.response_source.data = {
            "integrated": y,
            "nn": nn,
            "t": t,
        }

        for r in self.output_renderers:
            r.data_source.data = {
                "integrated": y,
                "nn": nn,
                "t": t,
            }

        nn_min = min(y.min(), nn.min())
        nn_min = 0.95 * nn_min if nn_min > 0 else 1.05 * nn_min

        nn_max = max(y.max(), nn.max())
        nn_max = 1.05 * nn_max if nn_max > 0 else 0.95 * nn_max

        self.timeseries_plot.extra_y_ranges["nn"].start = nn_min
        self.timeseries_plot.extra_y_ranges["nn"].end = nn_max

        self.timeseries_plot.xaxis.axis_label = (
            f"Time from {event_time:0.3f} [s]"
        )
        if event_type == "foreground":
            title = "Injected Event: "
            info = []
            for key, value in metadata.items():
                key = key.replace("_", " ")
                info.append(f"{key}={value:0.1f}")
            title += ", ".join(info)
        else:
            title = "Background Event"
        self.timeseries_plot.title.text = title

    def reset(self):
        self.configure_sources()
        self.timeseries_plot.title.text = "Click on an event to inspect"
        self.timeseries_plot.xaxis.axis_label = "Time [s]"
