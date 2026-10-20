import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from ledger.injections import (
    InterferometerResponseSet,
    shift_mask,
    waveform_class_factory,
)
from utils.preprocessing import BackgroundSnapshotter, BatchWhitener
from utils.streaming import StreamLayout, StreamOutputs

from plots.vizapp.infer.utils import get_strain_fname


@dataclass
class EventAnalysis:
    """Re-analysis of an event, with all times relative to the event.

    Attributes:
        availability_times:
            Time each network output could first be computed, i.e. when
            its newest input sample arrived, as in real-time observing.
        nn: Raw network outputs.
        integrated: Integrated network outputs.
        whitened_times: Time of each whitened strain sample.
        whitened: Whitened strain for each interferometer.
        detection_statistic:
            The integrated output corresponding to the analyzed event time.
    """

    availability_times: np.ndarray
    nn: np.ndarray
    integrated: np.ndarray
    whitened_times: np.ndarray
    whitened: dict[str, np.ndarray]
    detection_statistic: float


class EventAnalyzer:
    """
    Re-runs inference around events exactly as the pipeline did.

    Offline inference streams each background file from its start in
    batches, whitening each batch with a PSD from the data before it, so
    a detection statistic depends on where the batch boundaries fall.
    `analyze` replays those batches using `utils.streaming`, starting
    early enough that every batch it returns has full PSD history, or at
    the start of the stream, where the pipeline also started from an
    empty state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        strain_dir: Path,
        response_set: Path,
        psd_length: float,
        kernel_length: float,
        sample_rate: float,
        fduration: float,
        inference_sampling_rate: float,
        integration_length: float,
        batch_size: int,
        highpass: float,
        lowpass: float,
        fftlength: float | None,
        device: str,
        ifos: list[str],
    ):
        self.model = model
        self.whitener = BatchWhitener(
            kernel_length,
            sample_rate,
            inference_sampling_rate,
            batch_size,
            fduration,
            fftlength=fftlength,
            highpass=highpass,
            lowpass=lowpass,
            return_whitened=True,
        ).to(device)

        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to(device)

        self.stream_outputs = StreamOutputs(
            inference_sampling_rate, fduration, integration_length, psd_length
        )
        self.response_set = response_set
        self.strain_dir = strain_dir
        self.ifos = ifos
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.highpass = highpass
        self.lowpass = lowpass
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

    def stream_span(self, time: float) -> tuple[float, float]:
        """Span `[start, stop)` of the stream the pipeline analyzed
        `time` in."""
        _, t0, duration = get_strain_fname(self.strain_dir, time)
        return t0, t0 + duration

    def read_strain(
        self, ifo: str, start: int, stop: int, span: tuple[float, float]
    ) -> np.ndarray:
        """Strain for sample indices `[start, stop)` of the stream
        spanning `span`."""
        fname, _, _ = get_strain_fname(self.strain_dir, span[0])
        with h5py.File(fname, "r") as f:
            return f[ifo][start:stop]

    @cached_property
    def _injection_index(self):
        """`(injection_time, shift, duration)` of the waveform file.

        Read once and reused.
        """
        with h5py.File(self.response_set, "r") as f:
            times = f["parameters"]["injection_time"][:]
            shifts = f["parameters"]["shift"][:]
            duration = f.attrs["duration"]
        return times, shifts, duration

    def injections(self, start: float, stop: float, shifts: np.ndarray):
        """Every injection at `shifts` whose waveform overlaps GPS
        `[start, stop)`."""
        times, all_shifts, duration = self._injection_index
        mask = (times + duration >= start) & (times - duration <= stop)
        mask &= shift_mask(all_shifts, shifts)
        idx = np.flatnonzero(mask)
        return self.waveform_class.read_idx(self.response_set, idx)

    def _stream_layout(self, span, shifts) -> StreamLayout:
        size = round((span[1] - span[0]) * self.sample_rate)
        max_shift = int(max(shifts) * self.sample_rate)
        rate = self.stream_outputs.inference_sampling_rate
        stride = int(self.sample_rate / rate)
        return StreamLayout(size - max_shift, self.batch_size, stride)

    def _batch_range(self, stream_layout: StreamLayout, start: int, stop: int):
        """First and last batch indices covering samples `start` through
        `stop`, beginning early enough that all of them have a full
        snapshotter state before them, or the stream's start."""
        step_size = stream_layout.step_size
        history = math.ceil(self.snapshotter.state_size / step_size)
        first = start // step_size - history
        last = stop // step_size
        return max(first, 0), min(last, stream_layout.num_batches - 1)

    def infer(
        self,
        span: tuple[float, float],
        shifts: Sequence[float],
        stream_layout: StreamLayout,
        first: int,
        last: int,
        foreground: bool,
    ):
        """Run inference on batches `first` through `last` of the stream
        spanning `span`.

        Returns the raw outputs and the whitened strain, with the outputs
        and strain computed from padding removed.
        """
        start = stream_layout.batch_bounds(first)[0]
        stop = stream_layout.batch_bounds(last)[1]

        slice_layout = replace(stream_layout, size=stop - start)
        t0 = span[0] + start / self.sample_rate

        shift_sizes = [int(s * self.sample_rate) for s in shifts]
        size = slice_layout.num_batches * slice_layout.step_size
        X = np.zeros((len(self.ifos), size), dtype=np.float32)
        for i, (ifo, k) in enumerate(zip(self.ifos, shift_sizes, strict=True)):
            X[i, : slice_layout.size] = self.read_strain(
                ifo, start + k, stop + k, span
            )
        injected = X
        if foreground:
            t1 = t0 + size / self.sample_rate
            waveforms = self.injections(t0, t1, shifts)
            injected = waveforms.inject(X.copy(), t0)
        # whiten the second element with the PSD of the first
        # to match what we do in infer
        X = torch.tensor(np.stack([X, injected]), device=self.device)

        nn, whitened = [], []
        state_shape = (2, len(self.ifos), self.snapshotter.state_size)
        state = torch.zeros(state_shape, device=self.device)
        step_size = slice_layout.step_size
        with torch.no_grad():
            for b in range(slice_layout.num_batches):
                update = X[..., b * step_size : (b + 1) * step_size]
                x, state = self.snapshotter(update, state)
                kernels, w = self.whitener(x)
                nn.append(self.model(kernels)[:, 0].cpu().numpy())
                w = w[0].cpu().numpy()
                # consecutive batches' whitened data
                # overlap by kernel_size - stride samples
                if b > 0:
                    w = w[:, self.kernel_size - slice_layout.stride :]
                whitened.append(w)

        # drop what was computed from the final batch's padding
        nn = np.concatenate(nn)[: slice_layout.num_outputs]
        whitened = np.concatenate(whitened, axis=-1)
        whitened = whitened[:, : whitened.shape[1] - slice_layout.num_pad]
        return nn, whitened

    def analyze(
        self,
        time: float,
        shifts: Sequence[float],
        foreground: bool = False,
        window: tuple[float, float] = (-3, 5),
    ) -> EventAnalysis:
        """Re-run inference around `time`.

        Args:
            time: GPS time to analyze
            shifts: Timeslide shift of each interferometer.
            foreground: Whether to inject the waveforms at these shifts.
            window: Range of times relative to `time` to return.
        """
        stream_outputs = self.stream_outputs
        fduration = stream_outputs.fduration
        lag = stream_outputs.lag
        rate = stream_outputs.inference_sampling_rate
        span = self.stream_span(time)
        stream_layout = self._stream_layout(span, shifts)
        # index of the integrated output timestamped at `time`
        detection_idx = round((time + lag - span[0]) * rate) - 1

        # Batches covering the window and the detection output. which is
        # The detection output is available `lag` after `time`, so ensure
        # that's accounted for. `self._batch_range` also covers the
        # batches needed to fill the snapshotter prior to the window.
        # Whitening discards the newest fduration / 2 of data, so whitened
        # strain at the end of the window needs that much more.
        begin = time + min(window[0], lag)
        end = time + max(window[1], lag) + fduration / 2
        first, last = self._batch_range(
            stream_layout,
            int((begin - span[0]) * self.sample_rate),
            int((end - span[0]) * self.sample_rate),
        )
        nn, whitened = self.infer(
            span, shifts, stream_layout, first, last, foreground
        )
        first_output = first * stream_layout.batch_size

        integrated = stream_outputs.integrate(nn, first_output)
        detection_statistic = integrated[detection_idx - first_output]

        indices = first_output + np.arange(len(nn))
        available = stream_outputs.availability_time(span[0], indices) - time
        stop = stream_layout.batch_bounds(last)[1]
        samples = np.arange(stop - whitened.shape[-1], stop)
        whitened_times = (
            span[0] + samples / self.sample_rate - fduration / 2 - time
        )

        out_mask = (available >= window[0]) & (available <= window[1])
        whitened_mask = (whitened_times >= window[0]) & (
            whitened_times <= window[1]
        )
        return EventAnalysis(
            availability_times=available[out_mask],
            nn=nn[out_mask],
            integrated=integrated[out_mask],
            whitened_times=whitened_times[whitened_mask],
            whitened={
                ifo: w[whitened_mask]
                for ifo, w in zip(self.ifos, whitened, strict=True)
            },
            detection_statistic=detection_statistic,
        )

    def get_fft(self, analysis: EventAnalysis):
        """Amplitude spectrum of each interferometer's whitened strain."""
        ffts = {}
        for ifo in self.ifos:
            strain = analysis.whitened[ifo]
            ts = TimeSeries(strain, times=analysis.whitened_times)
            fft = ts.fft().crop(start=self.highpass, end=self.lowpass)
            freqs = fft.frequencies.value
            ffts[ifo] = np.abs(fft.value)

        return freqs, ffts

    def qscan(self, analysis: EventAnalysis):
        """Q-transform of each interferometer's whitened strain."""
        qscans = []
        for ifo in self.ifos:
            strain = analysis.whitened[ifo]
            ts = TimeSeries(strain, times=analysis.whitened_times)
            ts = ts.crop(-3, 3)
            qscan = ts.q_transform(
                logf=True, frange=(32, 1024), whiten=False, outseg=(-1, 1)
            )
            qscans.append(qscan)
        return qscans
