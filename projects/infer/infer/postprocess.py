import numpy as np
from ledger.events import EventSet
from utils.streaming import StreamOutputs


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
        **kwargs,
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

        self.stream_outputs = StreamOutputs(
            inference_sampling_rate,
            fduration,
            integration_window_length,
            psd_length,
        )
        self.shifts = shifts

        # timestamp of the first output kept after dropping the burn-in
        self.t0 = self.stream_outputs.timestamp(
            t0, self.stream_outputs.burn_in_size
        )
        self.cluster_window_size = int(
            inference_sampling_rate * cluster_window_length
        )

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

            if (val < window).any():
                # if there is a larger value,
                # move our index to it
                i += np.argmax(window) + 1
            else:
                # otherwise, this must be the largest value
                # in the full window around it, so record
                # the value and reset the index to be the
                # first value outside the current window
                events.append(val)
                t = self.t0 + i / self.stream_outputs.inference_sampling_rate
                times.append(t)
                i += window_size + 1

        # record all this info and some
        # metadata into a ledger object
        Tb = len(y) / self.stream_outputs.inference_sampling_rate
        events = np.array(events)
        times = np.array(times)
        shifts = np.ones((len(events), len(self.shifts))) * self.shifts
        return EventSet(events, times, shifts, Tb)

    def __call__(self, y: dict[str, np.ndarray | None] = {"detection_statistic": None}) -> EventSet:
        # in the case where we didn't perform
        # injections on this shift
        # just return an empty event set
        if y is None:
            return EventSet()
        y = y['detection_statistic']
        burn_in = self.stream_outputs.burn_in_size
        y = self.stream_outputs.integrate(y[burn_in:], first_output=burn_in)
        return self.cluster(y)


class MergerTimePostprocessor(Postprocessor):
    def __init__(
        self,
        kernel_length: float,
        *args,
        **kwargs
    ) -> None:
        """
        Postprocessor with merger time regression

        Args:
            kernel_length:
                Length of the window seen by the network in seconds
        """
        super().__init__(*args, **kwargs)
        self.inference_sampling_rate = kwargs['inference_sampling_rate']
        self.kernel_length = kernel_length
        self.integration_window_size = (
            int(kwargs['inference_sampling_rate'] * kwargs['integration_window_length']) + 1
        )

    def merger_time(self, pred_times: np.ndarray, i: int) -> float:
        """
        Estimate the merger time of an event found at index `i`
        of the integrated detection statistic.

        Takes the median of the predicted merger times over the
        trailing integration window. Note that this works only if:
            (kernel_length - left_pad - right_pad) * sampling_rate >
            integration_window_size / 2
        Otherwise, the median will be biased towards times from outside
        where the network was trained.
        """
        start = max(i - self.integration_window_size + 1, 0)
        return np.median(pred_times[start : i + 1])

    def cluster(self, y, pred_times) -> EventSet:
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

            if (val < window).any():
                # if there is a larger value,
                # move our index to it
                i += np.argmax(window) + 1
            else:
                # otherwise, this must be the largest value
                # in the full window around it, so record
                # the value and reset the index to be the
                # first value outside the current window
                events.append(val)
                times.append(self.merger_time(pred_times, i))
                i += window_size + 1

        # record all this info and some
        # metadata into a ledger object
        Tb = len(y) / self.stream_outputs.inference_sampling_rate
        events = np.array(events)
        times = np.array(times)
        shifts = np.ones((len(events), len(self.shifts))) * self.shifts
        return EventSet(events, times, shifts, Tb)

    def __call__(
        self, y: dict[str, np.ndarray] = {"detection_statistic": None, "heatmap": None}) -> EventSet:
        # in the case where we didn't perform
        # injections on this shift
        # just return an empty event set
        if y is None:
            return EventSet()

        burn_in = self.stream_outputs.burn_in_size
        y = {key: y[key][burn_in :] for key in y.keys()}

        y['detection_statistic'] = self.stream_outputs.integrate(y['detection_statistic'], first_output=burn_in)

        times = self.t0 + np.arange(len(y["detection_statistic"])) / self.inference_sampling_rate
        pred_times = (np.argmax(y["heatmap"], axis=-1) / y["heatmap"].shape[-1] * self.kernel_length + times)
        y = self.cluster(y['detection_statistic'], pred_times)
        return y