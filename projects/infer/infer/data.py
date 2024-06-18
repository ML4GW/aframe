import logging
import math
from contextlib import nullcontext
from typing import Optional
from zlib import adler32

import h5py
import numpy as np
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InterferometerResponseSet, waveform_class_factory
from ratelimiter import RateLimiter


class Sequence:
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
        self.background_fname = background_fname
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size
        self.rate = rate
        self.ifos = ifos

        if len(ifos) != len(shifts):
            raise ValueError(
                "Number of ifos must match number of shifts"
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

        # derive some properties from that metadata,
        # including come up with a semi-unique sequence
        # id derived from a hash of its most descriptive parts
        fingerprint = f"{self.t0}{self.duration}{shifts}".encode()
        self.id = adler32(fingerprint)
        self.shifts = [int(i * self.sample_rate) for i in shifts]
        self.stride = int(self.sample_rate / inference_sampling_rate)
        self.step_size = self.stride * batch_size
        # initialize some containers for handling during
        # the inference response callback
        self._started = {}
        self._done = {}
        self._sequences = {}
        size = len(self) * self.batch_size
        for i in range(2):
            seq_id = self.id + i
            self._started[seq_id] = False
            self._done[seq_id] = False
            self._sequences[seq_id] = np.zeros(size)

        # if there are no injections, we can mark
        # the injection sequence as started and done
        if self.injection_set is None:
            self._done[self.id + 1] = True
            self._started[self.id + 1] = True

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

    def __iter__(self):
        if self.rate is not None:
            # rate refers to the average number of requests
            # per second, but remember that each yield
            # corresponds to two inference requests. Rather
            # than splitting the period in half, we'll allow
            # two calls during a given period to help account
            # for the time required to e.g. serialize the data
            # into inference requests
            limiter = RateLimiter(max_calls=2, period=3.5 / self.rate)
        else:
            limiter = nullcontext()

        with h5py.File(self.background_fname, "r") as f:
            for i in range(len(self)):
                # if this is the last batch, we may need to pad it
                # to make it a full batch
                last = i == len(self) - 1
                # grab the current batch of updates from the file
                # and stack it into a 2D array
                x = []
                for ifo, shift in zip(self.ifos, self.shifts):
                    start = shift + i * self.step_size

                    # for all but last batch just
                    # increase by step size
                    end = start + self.step_size

                    # if this is the last batch
                    # and we need to pad it
                    # just step by the remainder
                    if last and self.remainder:
                        end = start + self.remainder

                    data = f[ifo][start:end]
                    # if this is the last batch
                    # possibly pad it to make it a full batch
                    if last:
                        data = np.pad(data, (0, self.num_pad), "constant")

                    x.append(data)
                x = np.stack(x).astype(np.float32)
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
                with limiter:
                    yield x, x_inj

    def __call__(self, y, request_id, sequence_id):
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

        # if both the background and foreground
        # sequences have completed, return them both,
        # slicing off the dummy data from the last batch
        if self.done:
            background = self._sequences[self.id][self.slice]
            foreground = None
            if self.injection_set is not None:
                foreground = self._sequences[self.id + 1][self.slice]
            return background, foreground

    def recover(self, foreground: EventSet) -> RecoveredInjectionSet:
        return RecoveredInjectionSet.recover(foreground, self.injection_set)
