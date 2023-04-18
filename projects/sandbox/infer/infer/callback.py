import logging
import time
from dataclasses import dataclass

import numpy as np

from bbhnet.analysis.ledger.events import TimeSlideEventSet


class SequenceNotStarted(Exception):
    pass


class ExistingSequence(Exception):
    pass


@dataclass
class Callback:
    """
    Callable class for handling asynchronous server
    responses for streaming inference across sequences
    of timeseries data. New sequences should be
    initialized by calling the `start_new_sequence`
    method before starting to submit requests. Only
    one sequence can be inferred upon at once.

    Once inference has completed for each sequence,
    it will be asynchronously convolved with a
    boxcar filter to perform local integration, then
    both the raw and integrated timeseries will be
    written to a file in `write_dir` with the filename
    `out_{start}-{stop}.hdf5`, where `start` and `stop`
    indicate the initial and final GPS timestamps of
    the sequence.

    Args:
        write_dir:
            Directory to which to save network outputs
            in HDF5 format.
        inference_sampling_rate:
            Rate at which to sample inference windows
            from the input timeseries. Represents the
            sample rate of the output timeseries.
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        window_length:
            Length of the window over which network
            outputs should be locally integrated,
            specified in seconds.
    """

    id: int
    sample_rate: float
    batch_size: float
    integration_window_length: float
    cluster_window_length: float
    fduration: float

    def __post_init__(self):
        # @alec can this be removed? I don't think it's used
        self._sequence = None
        self.offset = self.fduration / 2
        self.reset()

    def reset(self):
        self.start = self.num_steps = self.done = self._started = None

    @property
    def started(self):
        return self._started is not None and all(self._started.values())

    def wait(self):
        while not self.started:
            time.sleep(1e-3)

    def initialize(self, start: float, end: float):
        if self.start is not None:
            raise ExistingSequence(
                "Already doing inference on {} prediction "
                "long sequence with t0={}".format(self.num_steps, start)
            )

        duration = end - start
        num_predictions = duration * self.sample_rate
        num_steps = int(num_predictions // self.batch_size)
        num_predictions = int(num_steps * self.batch_size)

        self.background = np.zeros((num_predictions,))
        self.foreground = np.zeros((num_predictions,))

        self.start = start
        self.num_steps = num_steps
        self.done = {self.id: False, self.id + 1: False}
        self._started = {self.id: False, self.id + 1: False}

        return num_steps

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
        window_size = int(self.integration_window_length * self.sample_rate)
        window = np.ones((window_size,)) / window_size
        integrated = np.convolve(y, window, mode="full")
        return integrated[: -window_size + 1]

    def cluster(self, y) -> TimeSlideEventSet:
        # subtract off the time required for
        # the coalescence to exit the filter
        # padding and enter the input kernel
        # to the neural network
        t0 = self.start - self.fduration / 2

        # now subtract off the time required
        # for the integration window to
        # hit its maximum value
        t0 -= self.integration_window_length

        window_size = int(self.cluster_window_length * self.sample_rate / 2)
        i = np.argmax(y[:window_size])
        events, times = [], []
        while i < len(y):
            val = y[i]
            window = y[i + 1 : i + 1 + window_size]
            if any(val <= window):
                i += np.argmax(window) + 1
            else:
                events.append(val)
                t = t0 + i / self.sample_rate
                times.append(t)
                i += window_size + 1

        Tb = len(y) / self.sample_rate
        events = np.array(events)
        times = np.array(times)
        return TimeSlideEventSet(events, times, Tb)

    def postprocess(self, y):
        y = self.integrate(y)
        return self.cluster(y)

    def check_done(self, sequence_id, request_id):
        self.done[sequence_id] = (request_id + 1) >= self.num_steps
        return all(self.done.values())

    def __call__(self, y, request_id, sequence_id):
        # check to see if we've initialized a new
        # blank output array
        if self.start is None:
            raise SequenceNotStarted(
                "Must initialize sequence {} by calling "
                "`Callback.initialize` before submitting "
                "inference requests.".format(sequence_id)
            )

        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        y = y[:, 0]
        if sequence_id == self.id:
            self.background[start:stop] = y
        else:
            self.foreground[start:stop] = y
        self._started[sequence_id] = True

        if self.check_done(sequence_id, request_id):
            logging.debug(
                "Finished inference on {} steps-long sequence "
                "with t0 {}".format(self.num_steps, self.start)
            )
            background_events = self.postprocess(self.background)
            foreground_events = self.postprocess(self.foreground)
            logging.debug("Finished postprocessing")

            self.reset()
            return background_events, foreground_events
