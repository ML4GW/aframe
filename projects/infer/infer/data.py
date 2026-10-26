import logging
import re
import time
from abc import ABC, abstractmethod
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


def _throttle(deadline: float, interval: float) -> float:
    """Sleep until deadline, then return the next one interval later.

    Time already spent this iteration counts toward the interval, so a slow
    iteration shortens or eliminates the sleep.
    """
    now = time.monotonic()
    if now < deadline:
        time.sleep(deadline - now)
    return max(now, deadline) + interval


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


class BaseSequence(ABC):
    """Iterate over a segment of data, yielding (background, injected) batches
    and aggregating the inference responses returned through __call__.

    Subclasses supply source-specific setup and iteration.
    Everything else is shared.
    """

    def __init__(
        self,
        sequence_id: int,
        inference_sampling_rate: float,
        batch_size: int,
        rate: float | None = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.rate = rate
        self.id = sequence_id

        # Subclasses set sample_rate, size, t0, duration, and shifts.
        self._setup(**kwargs)

        self.stream_layout = StreamLayout(
            self.size - max(self.shifts),
            batch_size,
            int(self.sample_rate / inference_sampling_rate),
        )
        self._initialize_sequence_state()

    @abstractmethod
    def _setup(self, **kwargs):
        """Set sample_rate, size, t0, duration, and shifts."""

    @property
    @abstractmethod
    def inference_filename(self) -> str:
        """The file this sequence reads, for logging."""

    @property
    @abstractmethod
    def has_foreground(self) -> bool:
        """Whether a foreground sequence is produced."""

    def _initialize_sequence_state(self):
        self._started = {}
        self._done = {}
        self._sequences = {}
        size = len(self) * self.batch_size
        for i in range(2):
            seq_id = self.id + i
            self._started[seq_id] = False
            self._done[seq_id] = False
            self._sequences[seq_id] = np.zeros(size)

    @property
    def started(self):
        return all(self._started.values())

    @property
    def done(self):
        return all(self._done.values())

    @property
    def slice(self) -> slice:
        # drop the outputs computed from the final batch's padding
        return slice(self.stream_layout.num_outputs)

    def __len__(self):
        # includes the trailing excess that can't fill a full batch; we pad it
        # with zeros and slice the corresponding outputs back off afterward
        return self.stream_layout.num_batches

    def _pad_last_batch(self, data: np.ndarray):
        padding = ((0, 0), (0, self.stream_layout.num_pad))
        return np.pad(data, padding, "constant")

    def __call__(self, y, request_id, sequence_id):
        # insert the response at the right spot in the output array
        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self._sequences[sequence_id][start:stop] = y[:, 0]

        self._started[sequence_id] = True
        if request_id == len(self) - 1:
            self._done[sequence_id] = True

        # once both sequences finish, return them, slicing off the padded tail
        if self.done:
            background = self._sequences[self.id][self.slice]
            foreground = None
            if self.has_foreground:
                foreground = self._sequences[self.id + 1][self.slice]
            return background, foreground

    @abstractmethod
    def recover(self, foreground: EventSet):
        """Map recovered foreground events back to any injections."""


class Hdf5Sequence(BaseSequence):
    def __init__(
        self,
        sequence_id: int,
        background_fname: str,
        injection_set_fname: str | None,
        ifos: list[str],
        shifts: list[float],
        inference_sampling_rate: float,
        batch_size: int,
        rate: float | None = None,
    ):
        """
        Iterate over a background segment, performing timeshifts, optionally
        injecting waveforms, and aggregating the returned inference outputs.

        If the injection set is empty for this segment and shifts, inference
        on injections is skipped and `None` is returned for the foreground
        events.

        Args:
            sequence_id:
                ID for the background stream, with the foreground using the
                next one. Must be nonzero and unique.
            background_fname:
                Path to the background segment
            injection_set_fname:
                Path to the injection set file, or None for no injections
            ifos:
                Interferometer names
            shifts:
                Time shifts to apply to each interferometer
            inference_sampling_rate:
                Rate at which inference is performed
            batch_size:
                Number of inference requests to send to the model at once
            rate:
                Rate at which to send requests in Hz
        """
        logging.info("Initializing sequence")
        super().__init__(
            sequence_id=sequence_id,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            rate=rate,
            background_fname=background_fname,
            injection_set_fname=injection_set_fname,
            ifos=ifos,
            shifts=shifts,
        )
        # with no injections, mark the foreground sequence already complete
        if self.injection_set is None:
            self._done[self.id + 1] = True
            self._started[self.id + 1] = True

    def _setup(
        self,
        background_fname: str,
        injection_set_fname: str | None,
        ifos: list[str],
        shifts: list[float],
    ):
        self.background_fname = background_fname
        self.ifos = ifos

        if len(ifos) != len(shifts):
            raise ValueError(
                "Number of ifos must match number of shifts; "
                f"got {len(ifos)} ifos and {len(shifts)} shifts"
            )

        self.sample_rate, self.t0, self.size = read_metadata(
            background_fname, ifos[0]
        )
        self.duration = self.size / self.sample_rate

        # load injections up front. None means skip inference on injections
        self.injection_set = load_injections(
            injection_set_fname,
            ifos,
            self.t0,
            self.t0 + self.duration,
            shifts,
        )
        if self.injection_set is None:
            logging.info(
                f"No injections found for segment {background_fname} "
                f"and shifts {shifts}, skipping."
            )
        self.shifts = np.array([int(i * self.sample_rate) for i in shifts])

    @property
    def inference_filename(self):
        return self.background_fname

    @property
    def has_foreground(self):
        return self.injection_set is not None

    def __iter__(self):
        # rate is the average number of requests per second. Each yield is
        # two inference requests, so space yields 2 / rate seconds apart.
        interval = 2 / self.rate if self.rate is not None else 0.0
        deadline = time.monotonic()

        with h5py.File(self.background_fname, "r") as f:
            for i in range(len(self)):
                start, end, last = self.stream_layout.batch_bounds(i)
                x = [
                    f[ifo][start + shift : end + shift]
                    for ifo, shift in zip(self.ifos, self.shifts, strict=True)
                ]
                x = np.stack(x).astype(np.float32)
                x = self._pad_last_batch(x) if last else x

                # inject waveforms into a copy of the background, if any
                x_inj = None
                if self.injection_set is not None:
                    t0 = self.t0 + start / self.sample_rate
                    x_inj = self.injection_set.inject(x.copy(), t0)

                deadline = _throttle(deadline, interval)
                yield x, x_inj

    def recover(self, foreground: EventSet) -> RecoveredInjectionSet:
        return RecoveredInjectionSet.recover(foreground, self.injection_set)


class RnPSequence(BaseSequence):
    def __init__(
        self,
        sequence_id: int,
        injection_file: Path,
        channel: str,
        ifos: list[str],
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
    ):
        """
        Iterate over an injection frame that already contains injected signals.
        The same data is yielded as both "background" and "foreground", with
        no event recovery performed.

        Each sequence is one frame, so the PSD burn-in is performed on each
        frame. PSD length should be short enough that this doesn't impact
        performance much.

        Args:
            sequence_id:
                ID for the background stream, with the foreground using the
                next one. Must be nonzero and unique.
            injection_file:
                R&P injection frame file to analyze
            channel:
                Channel name within the frame, combined as {ifo}:{channel}
            ifos:
                Interferometer names
            sample_rate:
                Sample rate the data is resampled to for inference
            inference_sampling_rate:
                Rate at which inference is performed
            batch_size:
                Number of inference requests to send to the model at once
        """
        logging.info("Initializing sequence")
        self.sample_rate = sample_rate
        # always in-process: no server to throttle, so rate is always None
        super().__init__(
            sequence_id=sequence_id,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            rate=None,
            injection_file=injection_file,
            channel=channel,
            ifos=ifos,
        )

    def _setup(
        self,
        injection_file: Path,
        channel: str,
        ifos: list[str],
    ):
        self.ifos = ifos
        # Don't shift timeseries for R&P injections
        self.shifts = np.zeros(len(ifos), dtype=int)
        self.channels = [f"{ifo}:{channel}" for ifo in ifos]

        self.injection_file = Path(injection_file)
        match = FNAME_RE.search(self.injection_file.name)
        if not match:
            raise ValueError(
                f"{self.injection_file.name} does not match expected pattern"
            )
        self.t0 = int(match.group("start"))
        self.duration = int(match.group("duration"))
        self.size = int(self.duration * self.sample_rate)

        injected = TimeSeriesDict.read(
            self.injection_file, channels=self.channels
        ).resample(self.sample_rate)
        self.timeseries = np.stack(
            [injected[ch].value for ch in self.channels]
        ).astype(np.float32)

    @property
    def inference_filename(self):
        return self.injection_file.name

    @property
    def has_foreground(self):
        return True

    def __iter__(self):
        for i in range(len(self)):
            start, end, last = self.stream_layout.batch_bounds(i)
            x_inj = self.timeseries[:, start:end]
            x_inj = self._pad_last_batch(x_inj) if last else x_inj
            # yield the same data as background and foreground
            yield x_inj, x_inj

    def recover(self, foreground: EventSet) -> EventSet:
        return foreground


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
        ifos: List of interferomters
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

        Unlike `Hdf5Sequence`, which streams one shift, this batches the
        shifts of a segment so that they share each read and model call.

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
