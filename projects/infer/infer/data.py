from contextlib import nullcontext
from typing import Optional
from zlib import adler32

import h5py
import numpy as np
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import LigoResponseSet
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
        self.background_fname = background_fname
        self.ifos = ifos
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size
        self.rate = rate

        # read some of the metadata from our background file
        with h5py.File(background_fname, "r") as f:
            dataset = f[ifos[0]]
            self.size = len(dataset)
            self.sample_rate = 1 / dataset.attrs["dx"]
            self.t0 = dataset.attrs["x0"]
            self.duration = self.size / self.sample_rate

        # use this to load in our injections up front
        self.injection_set = LigoResponseSet.read(
            injection_set_fname,
            start=self.t0,
            end=self.t0 + self.duration,
            shifts=shifts,
        )

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

    @property
    def started(self):
        return all(self._started.values())

    @property
    def done(self):
        return all(self._done.values())

    def __len__(self):
        return (self.size - max(self.shifts)) // self.step_size

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
                # grab the current batch of updates from the file
                # and stack it into a 2D array
                x = []
                for ifo, shift in zip(self.ifos, self.shifts):
                    start = shift + i * self.step_size
                    x.append(f[ifo][start : start + self.step_size])
                x = np.stack(x).astype(np.float32)

                # injection any waveforms into a copy of the background
                offset = i * self.batch_size / self.inference_sampling_rate
                x_inj = self.injection_set.inject(x.copy(), self.t0 + offset)

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
        # sequences have completed, return them both
        if self.done:
            return tuple(self._sequences[self.id + i] for i in range(2))

    def recover(self, foreground: EventSet) -> RecoveredInjectionSet:
        return RecoveredInjectionSet.recover(foreground, self.injection_set)
