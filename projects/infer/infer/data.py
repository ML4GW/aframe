import logging
import math
import re
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from zlib import adler32

import h5py
import numpy as np
from ratelimiter import RateLimiter
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


class BaseSequence(ABC):
    def __init__(
        self,
        inference_sampling_rate: float,
        batch_size: int,
        rate: float | None = None,
        **kwargs,
    ):
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size

        # Subclasses provide source-specific setup for metadata and data.
        self._setup(**kwargs)

        self.stride = int(self.sample_rate / inference_sampling_rate)
        self.step_size = self.stride * batch_size

        # derive some properties from that metadata,
        # including come up with a semi-unique sequence
        # id derived from a hash of its most descriptive parts
        fingerprint = f"{self.t0}{self.duration}{self.shifts}".encode()
        self.id = adler32(fingerprint)
        self._initialize_sequence_state()

        # rate refers to the average number of requests
        # per second, but remember that each yield
        # corresponds to two inference requests. Rather
        # than splitting the period in half, we'll allow
        # two calls during a given period to help account
        # for the time required to e.g. serialize the data
        # into inference requests
        self.limiter = (
            RateLimiter(max_calls=2, period=3.5 / rate)
            if rate
            else nullcontext()
        )

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

    @abstractmethod
    def _setup(self, **kwargs):
        """Subclasses must set sample_rate, size, t0, duration, and shifts."""
        pass

    @property
    @abstractmethod
    def inference_filenames(self):
        pass

    @property
    @abstractmethod
    def has_foreground(self):
        pass

    @property
    def started(self):
        return all(self._started.values())

    @property
    def done(self):
        return all(self._done.values())

    @property
    def remainder(self):
        # the number of remaining data points not filling a full batch
        return (self.size - max(self.shifts)) % self.step_size

    @property
    def num_pad(self):
        # the number of zeros we need to pad the last batch
        # to make it a full batch
        return (self.step_size - self.remainder) % self.step_size

    @property
    def slice(self) -> slice:
        """
        The number of inference requests we need to slice
        off the end of the sequences to remove
        the dummy data from the last batch
        """

        # if num_pad is 0 don't slice anything
        num_slice = self.num_pad // self.stride
        end = -num_slice if num_slice else None
        return slice(end)

    def __len__(self):
        # this include excess data at end of sequence that can't
        # be used for a full batch. We'll end up padding it
        # with zeros to make it a full batch and
        # slicing off the actual useful inference requests
        # corresponding to the excess
        return math.ceil((self.size - max(self.shifts)) / self.step_size)

    def _finalize_sequence(self, has_foreground: bool = True):
        if not self.done:
            return None

        # if both the background and foreground
        # sequences have completed, return them both,
        # slicing off the dummy data from the last batch
        background = self._sequences[self.id][self.slice]
        foreground = None
        if has_foreground:
            foreground = self._sequences[self.id + 1][self.slice]
        return background, foreground

    def _record_response(self, y, request_id, sequence_id):
        # insert the response at the appropriate
        # spot in the corresponding output array
        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self._sequences[sequence_id][start:stop] = y[:, 0]

        # indicate that the first response for
        # this sequence has returned, and possibly
        # that the last one has returned as well
        self._started[sequence_id] = True
        if request_id == len(self) - 1:
            self._done[sequence_id] = True

    def __call__(self, y, request_id, sequence_id):
        self._record_response(y, request_id, sequence_id)
        return self._finalize_sequence(has_foreground=self.has_foreground)

    def _get_data_indices(self, batch_idx: int, shift: int = 0):
        # if this is the last batch, we may need to pad it
        # to make it a full batch
        last = batch_idx == len(self) - 1
        # grab the current batch of updates from the file
        # and stack it into a 2D array
        start = batch_idx * self.step_size + shift

        # for all but last batch just
        # increase by step size
        end = start + self.step_size

        # if this is the last batch
        # and we need to pad it
        # just step by the remainder
        if last and self.remainder:
            end = start + self.remainder

        return start, end, last

    def _pad_last_batch(self, data: np.ndarray):
        return np.pad(data, ((0, 0), (0, self.num_pad)), "constant")


class Hdf5Sequence(BaseSequence):
    def __init__(
        self,
        background_fname: str,
        injection_set_fname: str,
        ifos: list[str],
        shifts: list[float],
        inference_sampling_rate: float,
        batch_size: int,
        rate: Optional[float] = None,
    ):
        """
        Object used for iterating over a segment of data,
        performing timeshifts, optionally injecting waveforms, and
        aggregating the returned inference outputs.

        If the injection set is empty for this given
        segment and shifts, infernece on injections will be skipped,
        and `None` will be returned for the foreground events.

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

        # load in our injections up front
        # if there are no injections for
        # this shift, set it to None so
        # we don't run inference on injections

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
    def inference_filenames(self):
        return [self.background_fname]

    @property
    def has_foreground(self):
        return self.injection_set is not None

    def __iter__(self):
        with h5py.File(self.background_fname, "r") as f:
            for i in range(len(self)):
                x = []
                for ifo, shift in zip(self.ifos, self.shifts, strict=True):
                    start, end, last = self._get_data_indices(i, shift)
                    data = f[ifo][start:end]
                    x.append(data)

                x = np.stack(x).astype(np.float32)
                x = self._pad_last_batch(x) if last else x

                # if there are any injections for this shift,
                # inject waveforms into a copy of the background
                x_inj = None
                offset = i * self.batch_size / self.inference_sampling_rate
                if self.injection_set is not None:
                    x_inj = self.injection_set.inject(
                        x.copy(), self.t0 + offset
                    )

                # return the two sets of updates, possibly
                # rate limited if we specified a max rate
                with self.limiter:
                    yield x, x_inj

    def recover(self, foreground: EventSet) -> RecoveredInjectionSet:
        return RecoveredInjectionSet.recover(foreground, self.injection_set)


class RnPSequence(BaseSequence):
    def __init__(
        self,
        injection_files: list[Path],
        channel: str,
        ifos: list[str],
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        rate: Optional[float] = None,
    ):
        """
        Object used for iterating over a segment of data,
        performing timeshifts, optionally injecting waveforms, and
        aggregating the returned inference outputs.

        If the injection set is empty for this given
        segment and shifts, infernece on injections will be skipped,
        and `None` will be returned for the foreground events.

        Args:
            injection_files:
                List of R&P injection files to be analyzed.
            channel:
                Name of the channel within the frame file
            ifos:
                Interferometer names
            sample_rate:
                Sample rate that data will be resampled to for inference
            inference_sampling_rate:
                Rate at which inference is performed
            batch_size:
                Number of inference requests to send to the model at once
            rate:
                Rate at which to send requests in Hz
        """
        logging.info("Initializing sequence")

        self.sample_rate = sample_rate
        super().__init__(
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            rate=rate,
            injection_files=injection_files,
            channel=channel,
            ifos=ifos,
        )

    def _setup(
        self,
        injection_files: list[Path],
        channel: str,
        ifos: list[str],
    ):
        self.ifos = ifos

        # Don't shift timeseries for R&P injections
        self.shifts = np.zeros(len(ifos), dtype=int)
        self.channels = [f"{ifo}:{channel}" for ifo in ifos]

        if not injection_files:
            raise ValueError("Must provide at least one injection file")

        self.injection_files = sorted(injection_files)

        matches = [
            FNAME_RE.search(fname.name) for fname in self.injection_files
        ]
        if not all(matches):
            raise ValueError(
                "All injection files must match expected name pattern"
            )

        starts = [int(match.group("start")) for match in matches]
        durations = [int(match.group("duration")) for match in matches]

        self.t0 = min(starts)
        self.duration = sum(durations)
        self.size = int(self.duration * self.sample_rate)
        self.timeseries = np.zeros((len(ifos), self.size), dtype=np.float32)

        # Load and resample data from each file
        for file, start, duration in zip(
            injection_files, starts, durations, strict=True
        ):
            start_idx = int(self.sample_rate * (start - self.t0))
            end_idx = start_idx + int(self.sample_rate * duration)
            injected = TimeSeriesDict.read(file, channels=self.channels)
            injected = injected.resample(self.sample_rate)
            self.timeseries[:, start_idx:end_idx] = np.stack(
                [injected[ch].value for ch in self.channels]
            )

    @property
    def inference_filenames(self):
        return [fname.name for fname in self.injection_files]

    @property
    def has_foreground(self):
        return True

    def __iter__(self):
        for i in range(len(self)):
            start, end, last = self._get_data_indices(i)

            x_inj = self.timeseries[:, start:end]
            x_inj = self._pad_last_batch(x_inj) if last else x_inj

            # yield the same data twice, as we normally
            # expect to pass background and foreground
            with self.limiter:
                yield x_inj, x_inj

    def recover(self, foreground: EventSet) -> EventSet:
        return foreground
