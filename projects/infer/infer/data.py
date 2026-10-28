import re
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
from gwpy.timeseries import TimeSeriesDict
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InterferometerResponseSet, waveform_class_factory
from utils.streaming import StreamLayout

from infer.postprocess import Postprocessor

FNAME_PATTERNS = {
    "prefix": "[a-zA-Z0-9_:-]+",
    "start": "[0-9]{10}",
    "duration": "[1-9][0-9]*",
    "suffix": "(gwf)|(hdf5)|(h5)",
}
FNAME_GROUPS = {k: f"(?P<{k}>{v})" for k, v in FNAME_PATTERNS.items()}
FNAME_PATTERN = "{prefix}-{start}-{duration}.{suffix}".format(**FNAME_GROUPS)
FNAME_RE = re.compile(FNAME_PATTERN)


def read_metadata(fname: str, ifo: str) -> tuple[float, float, int]:
    """Sample rate, start time and number of samples of one interferometer's
    data in a background file."""
    with h5py.File(fname, "r") as f:
        dataset = f[ifo]
        return 1 / dataset.attrs["dx"], dataset.attrs["x0"], len(dataset)


def load_injections(
    fname: str | None,
    ifos: list[str],
    start: float,
    end: float,
    shifts: list[float],
) -> InterferometerResponseSet | None:
    """The injections in `fname` between `start` and `end` at `shifts`, or
    None if there's no file or no injections."""
    if fname is None:
        return None
    cls = waveform_class_factory(
        ifos, InterferometerResponseSet, "ResponseSet"
    )
    injection_set = cls.read(fname, start=start, end=end, shifts=shifts)
    return injection_set if len(injection_set) else None


def read_shifted(
    f: h5py.File,
    ifos: list[str],
    shifts: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """
    Read all timeshifts from a file, with one read per interferometer.

    Args:
        f: h5py.File object to read from
        ifos: List of interferometers
        shifts:
            Arrays of shifts, in units of samples, of shape [S, C], with
            one row per shift and one column per detector
        start: Unshifted index of the first sample
        end:
            Unshifted index of the final sample plus one, so that samples
            span [start, end).

    Return:
        Array of shape [S, C, end - start]: every shift for all detectors.
    """
    span = end - start
    shifted_timeseries = []
    for i, ifo in enumerate(ifos):
        ifo_shifts = shifts[:, i]
        min_shift = min(ifo_shifts)
        shifted_start = start + min_shift
        shifted_end = end + max(ifo_shifts)
        full_timeseries = f[ifo][shifted_start:shifted_end]
        offsets = ifo_shifts - min_shift
        shifted_timeseries.append(
            np.stack(
                [full_timeseries[s : span + s] for s in offsets],
            )
        )
    return np.stack(shifted_timeseries, axis=1)


class ShiftBatchSequence:
    """Stream one background file at several shifts at once."""

    def __init__(
        self,
        background_fname: str,
        waveform_fnames: list[str | None],
        ifos: list[str],
        shifts: list[list[float]],
        inference_sampling_rate: float,
        batch_size: int,
    ):
        """
        Iterate over a background segment at several timeshifts at once,
        injecting waveforms into a copy of each shift that has any.

        Each step yields one array of streams: the background at every
        shift, followed by the injected copy of each shift in
        `injected_idx`.

        `psd_source` says which stream's PSD each stream is
        whitened with, and `results` turns the outputs of all the steps
        into each shift's events. A shift without injections gets an empty
        set of recovered injections.

        Args:
            background_fname:
                Path to the background segment
            waveform_fnames:
                One injection set file per shift, or None for a shift
                without one
            ifos:
                Interferometer names
            shifts:
                One shift per branch, in seconds, with one value per
                interferometer
            inference_sampling_rate:
                Rate at which inference is performed
            batch_size:
                Number of outputs each stream produces per step
        """
        self.background_fname = background_fname
        self.ifos = ifos
        self.sample_rate, self.t0, self.size = read_metadata(
            background_fname, ifos[0]
        )
        duration = self.size / self.sample_rate

        self.injection_sets = [
            load_injections(fname, ifos, self.t0, self.t0 + duration, shift)
            for fname, shift in zip(waveform_fnames, shifts, strict=True)
        ]
        self.injected_idx = [
            i
            for i, injections in enumerate(self.injection_sets)
            if injections is not None
        ]

        self.shifts = (np.array(shifts) * self.sample_rate).astype(int)

        self.stream_layout = StreamLayout(
            self.size - np.max(self.shifts),
            batch_size,
            int(self.sample_rate / inference_sampling_rate),
        )

    def __len__(self) -> int:
        return self.stream_layout.num_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        with h5py.File(self.background_fname, "r") as f:
            for i in range(len(self)):
                start, end, last = self.stream_layout.batch_bounds(i)
                shifted_batch = read_shifted(
                    f, self.ifos, self.shifts, start, end
                )
                x = shifted_batch.astype(np.float32)

                if last:
                    padding = ((0, 0), (0, 0), (0, self.stream_layout.num_pad))
                    x = np.pad(x, padding, "constant")

                x_inj = []
                t0 = self.t0 + start / self.sample_rate
                for s in self.injected_idx:
                    x_inj.append(
                        self.injection_sets[s].inject(x[s].copy(), t0)
                    )

                if len(x_inj) > 0:
                    x = np.concatenate([x, np.stack(x_inj)], axis=0)
                yield x

    def recover(
        self, foreground: EventSet, index: int
    ) -> RecoveredInjectionSet:
        """Match the foreground events of the shift at `index` to its
        injections"""
        return RecoveredInjectionSet.recover(
            foreground, self.injection_sets[index]
        )

    @property
    def psd_source(self) -> list[int]:
        """For each stream, the index of the stream used to estimate
        the PSD used for whitening."""
        # The background streams come first and each uses its own PSD. The
        # injected streams come after, each using the PSD of its shift's
        # background.
        num_shifts = len(self.shifts)
        return list(range(num_shifts)) + list(self.injected_idx)

    def results(
        self, outputs: np.ndarray, postprocessors: list[Postprocessor]
    ) -> list[
        tuple[EventSet, RecoveredInjectionSet, np.ndarray, np.ndarray | None]
    ]:
        """
        Transform the raw outputs of each stream into events.

        Args:
            outputs:
                Array of `(N, num_batches * batch_size)` model outputs,
                one row per input stream. Rows are arranged in the order
                that `__iter__` stacks them.
            postprocessors:
                A list of postprocessors for each shift, arranged in
                the order of the shifts.

        Returns:
            A list of tuples, one tuple per shift. Each tuple holds
            `(background events, recovered injections, background outputs,
              foreground outputs or None)`.
        """
        # Trim the outputs from the padded final batch
        outputs = outputs[:, : self.stream_layout.num_outputs]

        # Split the rows into each shift's background and injected outputs.
        # The first S rows are the background streams, and row S + j is the
        # injected stream of shift injected_idx[j].
        num_shifts = len(self.shifts)
        background_outputs, foreground_outputs = (
            outputs[:num_shifts],
            outputs[num_shifts:],
        )

        # Postprocess each shift's outputs into events, and match its
        # injected events to that shift's injections.
        results = []
        for s, postprocessor in enumerate(postprocessors):
            background_ts = background_outputs[s]
            background = postprocessor(background_ts)
            foreground_ts = None
            if s in self.injected_idx:
                idx = self.injected_idx.index(s)
                foreground_ts = foreground_outputs[idx]
            foreground = self.recover(postprocessor(foreground_ts), s)

            results.append(
                (
                    background,
                    foreground,
                    background_ts,
                    foreground_ts,
                )
            )

        return results


class RnPStreamSequence:
    """Stream one R&P frame, whose data already contains its injections."""

    def __init__(
        self,
        injection_file: Path,
        channel: str,
        ifos: list[str],
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
    ):
        """
        Args:
            injection_file: R&P file matching "{prefix}-{start}-{duration}.gwf"
            channel:
                Channel name within the frame, which is read as
                f"{ifo}:{channel}"
            ifos: Interferometer names
            sample_rate:
                The sampling rate in Hz to which to resample the data in the
                frame file
            inference_sampling_rate: The rate of the model's outputs in Hz
            batch_size: Number of outputs the stream produces per step
        """
        self.injection_file = Path(injection_file)
        match = FNAME_RE.search(self.injection_file.name)
        if not match:
            raise ValueError(
                f"{self.injection_file.name} does not match expected pattern"
            )
        self.t0 = int(match.group("start"))
        duration = int(match.group("duration"))
        self.size = int(duration * sample_rate)

        channels = [f"{ifo}:{channel}" for ifo in ifos]
        injected = TimeSeriesDict.read(
            self.injection_file, channels=channels
        ).resample(sample_rate)
        self.timeseries = np.stack(
            [injected[ch].value for ch in channels]
        ).astype(np.float32)

        self.stream_layout = StreamLayout(
            self.size, batch_size, int(sample_rate / inference_sampling_rate)
        )

    def __len__(self) -> int:
        return self.stream_layout.num_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate through the timeseries, yielding batches. The frame is a
        single stream, so add a leading axis of size one.
        """
        for i in range(len(self)):
            start, end, last = self.stream_layout.batch_bounds(i)
            x = self.timeseries[:, start:end]
            if last:
                padding = ((0, 0), (0, self.stream_layout.num_pad))
                x = np.pad(x, padding, "constant")
            yield x[None]

    @property
    def psd_source(self) -> list[int]:
        return [0]

    def results(
        self, outputs: np.ndarray, postprocessors: list[Postprocessor]
    ) -> list[tuple[EventSet, EventSet, np.ndarray, np.ndarray]]:
        # Trim the outputs computed from the padding of the final step.
        timeseries = outputs[0, : self.stream_layout.num_outputs]
        events = postprocessors[0](timeseries)
        return [(events, events, timeseries, timeseries)]
