import logging
import math
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from zlib import adler32

import h5py
import numpy as np
from gwpy.timeseries import TimeSeriesDict
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InterferometerResponseSet, waveform_class_factory

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


class BaseSequence(ABC):
    """Iterate over a segment of data, yielding (background, injected) batches
    and aggregating the inference responses returned through __call__.

    Subclasses supply source-specific setup and iteration.
    Everything else is shared.
    """

    def __init__(
        self,
        inference_sampling_rate: float,
        batch_size: int,
        rate: float | None = None,
        **kwargs,
    ):
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size
        self.rate = rate

        # Subclasses set sample_rate, size, t0, duration, and shifts.
        self._setup(**kwargs)

        self.stride = int(self.sample_rate / inference_sampling_rate)
        self.step_size = self.stride * batch_size

        # a semi-unique sequence id from a hash of the descriptive metadata
        fingerprint = f"{self.t0}{self.duration}{self.shifts}".encode()
        self.id = adler32(fingerprint)
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
    def remainder(self):
        # number of remaining data points not filling a full batch
        return (self.size - max(self.shifts)) % self.step_size

    @property
    def num_pad(self):
        # zeros needed to pad the last batch to a full batch
        return (self.step_size - self.remainder) % self.step_size

    @property
    def slice(self) -> slice:
        # inference requests to slice off the end to drop the padded dummy data
        num_slice = self.num_pad // self.stride
        end = -num_slice if num_slice else None
        return slice(end)

    def __len__(self):
        # includes the trailing excess that can't fill a full batch; we pad it
        # with zeros and slice the corresponding outputs back off afterward
        return math.ceil((self.size - max(self.shifts)) / self.step_size)

    def _get_data_indices(self, batch_idx: int, shift: int = 0):
        last = batch_idx == len(self) - 1
        start = batch_idx * self.step_size + shift
        end = start + self.step_size
        # the last batch steps only by the remainder before padding
        if last and self.remainder:
            end = start + self.remainder
        return start, end, last

    def _pad_last_batch(self, data: np.ndarray):
        return np.pad(data, ((0, 0), (0, self.num_pad)), "constant")

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
        background_fname: str,
        injection_set_fname: str,
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
            background_fname:
                Path to the background segment
            injection_set_fname:
                Path to the injection set file
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
        injection_set_fname: str,
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

        # read some of the metadata from our background file
        with h5py.File(background_fname, "r") as f:
            dataset = f[ifos[0]]
            self.size = len(dataset)
            self.sample_rate = 1 / dataset.attrs["dx"]
            self.t0 = dataset.attrs["x0"]
            self.duration = self.size / self.sample_rate

        # load injections up front. None means skip inference on injections
        cls = waveform_class_factory(
            ifos,
            InterferometerResponseSet,
            "ResponseSet",
        )
        injection_set = cls.read(
            injection_set_fname,
            start=self.t0,
            end=self.t0 + self.duration,
            shifts=shifts,
        )
        if len(injection_set) == 0:
            logging.info(
                f"No injections found in {injection_set_fname} "
                f"for segment {background_fname} and "
                f"shifts {shifts}, skipping."
            )
            injection_set = None

        self.injection_set = injection_set
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
                x = []
                for ifo, shift in zip(self.ifos, self.shifts, strict=True):
                    start, end, last = self._get_data_indices(i, shift)
                    x.append(f[ifo][start:end])
                x = np.stack(x).astype(np.float32)
                x = self._pad_last_batch(x) if last else x

                # inject waveforms into a copy of the background, if any
                x_inj = None
                offset = i * self.batch_size / self.inference_sampling_rate
                if self.injection_set is not None:
                    x_inj = self.injection_set.inject(
                        x.copy(), self.t0 + offset
                    )

                deadline = _throttle(deadline, interval)
                yield x, x_inj

    def recover(self, foreground: EventSet) -> RecoveredInjectionSet:
        return RecoveredInjectionSet.recover(foreground, self.injection_set)


class RnPSequence(BaseSequence):
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
        Iterate over an injection frame that already contains injected signals.
        The same data is yielded as both "background" and "foreground", with
        no event recovery performed.

        Each sequence is one frame, so the PSD burn-in is performed on each
        frame. PSD length should be short enough that this doesn't impact
        performance much.

        Args:
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
            start, end, last = self._get_data_indices(i)
            x_inj = self.timeseries[:, start:end]
            x_inj = self._pad_last_batch(x_inj) if last else x_inj
            # yield the same data as background and foreground
            yield x_inj, x_inj

    def recover(self, foreground: EventSet) -> EventSet:
        return foreground
