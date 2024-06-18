from typing import Optional

import numpy as np
from ledger.events import EventSet


class Postprocessor:
    def __init__(
        self,
        t0: float,
        shifts: list[float],
        psd_length: float,
        fduration: float,
        inference_sampling_rate: float,
        integration_window_length: float,
        cluster_window_length: float,
    ) -> None:
        """
        Postprocessor object for converting timeseries
        of inference outputs into a set of events.

        Args:
            t0:
                GPS time of the start of the data segment
            shifts:
                Time shifts to applied to each interferometer
            psd_length:
                Length of the PSD data used in inference in seconds
            fduration:
                Duration of the whitening filter used in seconds
            inference_sampling_rate:
                Rate at which inference was performed
            integration_window_length:
                Length of the integration window in seconds
            cluster_window_length:
                Length of the clustering window in seconds
        """

        self.inference_sampling_rate = inference_sampling_rate
        self.shifts = shifts

        # offset our initial time both by the psd data
        # that we're going to slough off as well as by
        # the filter settle-in and integration time
        self.t0 = t0 + psd_length - fduration / 2 - integration_window_length
        self.offset = int(psd_length * inference_sampling_rate)

        # convert our window lengths to sample units
        self.integration_window_size = int(
            inference_sampling_rate * integration_window_length
        )
        self.cluster_window_size = int(
            inference_sampling_rate * cluster_window_length
        )

    def integrate(self, y: np.ndarray) -> np.ndarray:
        """
        Convolve predictions with boxcar filter
        to get local integration, slicing off of
        the last values so that timeseries represents
        integration of _past_ data only.
        "Full" convolution means first few samples are
        integrated with 0s, so will have a lower magnitude
        than they technically should.
        """
        window_size = self.integration_window_size
        window = np.ones((window_size,)) / window_size
        integrated = np.convolve(y, window, mode="full")
        return integrated[: -window_size + 1]

    def cluster(self, y) -> EventSet:
        # initial our search index to be in the first
        # half window of the timeseries. Then all we
        # need to know is whether there's a louder event
        # in the half window _after_ it to know that it's
        # the largest within the full window
        window_size = int(self.cluster_window_size // 2)
        i = np.argmax(y[:window_size])

        events, times = [], []
        while i < len(y):
            # check if there are any values in the next half
            # window which are larger than the current index
            val = y[i]
            window = y[i + 1 : i + 1 + window_size]

            if (val <= window).any():
                # if there is a larger value,
                # move our index to it
                i += np.argmax(window) + 1
            else:
                # otherwise, this must be the largest value
                # in the full window around it, so record
                # the value and reset the index to be the
                # first value outside the current window
                events.append(val)
                t = self.t0 + i / self.inference_sampling_rate
                times.append(t)
                i += window_size + 1

        # record all this info and some
        # metadata into a ledger object
        Tb = len(y) / self.inference_sampling_rate
        events = np.array(events)
        times = np.array(times)
        shifts = np.ones((len(events), len(self.shifts))) * self.shifts
        return EventSet(events, times, shifts, Tb)

    def __call__(self, y: Optional[np.ndarray] = None) -> EventSet:
        # in the case where we didn't perform
        # injections on this shift
        # just return an empty event set
        if y is None:
            return EventSet()
        y = y[self.offset :]
        y = self.integrate(y)
        y = self.cluster(y)
        return y
