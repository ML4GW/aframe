from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np

from bbhnet.io.timeslides import Segment

SECONDS_IN_YEAR = 31556952


@dataclass
class Distribution:
    dataset: str

    def __post_init__(self):
        self.Tb = 0
        self.fnames = []

    def write(self, path: Path):
        raise NotImplementedError

    def update(self, x: np.ndarray, t: np.ndarray):
        """Update this distribution to reflect new data"""

        raise NotImplementedError

    def nb(self, threshold: float):
        """Number of events in this distribution above a threshold"""

        raise NotImplementedError

    def far(self, threshold: float) -> float:
        """Compute the false alarm rate in units of yrs^{-1}"""

        nb = self.nb(threshold)
        return SECONDS_IN_YEAR * nb / self.Tb

    def significance(self, threshold: float, T: float) -> float:
        """see https://arxiv.org/pdf/1508.02357.pdf, eq. 17

        Represents the likelihood that at least one event
        with detection statistic value `threshold` will occur
        after observing this distribution for a period `T`.

        Args:
            threshold: The detection statistic to compare against
            T:
                The length of the analysis period in which the
                detection statistic was measured, in seconds
        """

        nb = self.nb(threshold)
        return 1 - np.exp(-T * (1 + nb) / self.Tb)

    def characterize_events(
        self,
        segment: Union["Segment", Tuple[np.ndarray, np.ndarray]],
        event_times: Union[float, Iterable[float]],
        window_length: float = 1,
        metric: str = "far",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Characterize known events in a segment according

        For a segment containing known events at times
        `event_times`, characterize the detection of that event
        by the dataset value associated with this distribution.
        This means computing the indicated `metric` for each
        timestep which falls within `window_length` of each
        event trigger time, then return the timeseries of metrics
        as well as latencies from the indicated trigger time.

        Args:
            segment: The Segment object to analyze, or a (y,t) tuple
            event_times:
                GPS times at which events are known to occur
                within the given segment. If only a single float
                is provided, the returned arrays wiill be 1-dimensional.
                Otherwise, the returned arrays will contain
                characterizations of each event stacked along the
                0th dimension.
            window_length:
                The period of time after each event trigger for
                which to provide characterizations.
            metric:
                The metric to use for characterization. Must be
                either `"far"` or `"significance"`.
        Returns:
            Timeseries of indicated metric values. If `event_times`
                was provided as a float, this will be a 1D timeseries.
                Otherwise, it will have a 0th dimension of size
                `len(event_times)`.
            Timeseries of latencies of each metric value from the
                corresponding event trigger time. Will have the same
                shape as the metric value return array.
        """
        if metric not in ("far", "significance"):
            raise ValueError(
                f"Can't characterize with unknown metric {metric}"
            )

        # duck-typing check on whether there are
        # multiple events in the segment or just the one.
        # Even if there's just one but it's passed as an
        # iterable, we'll record return a 2D array, otherwise
        # just return 1D
        try:
            event_iter = iter(event_times)
            single_event = False
        except TypeError:
            event_iter = iter([event_times])
            single_event = True

        try:
            y, t = segment.load(self.dataset)
        except AttributeError:
            y, t = segment

        sample_rate = 1 / (t[1] - t[0])
        window_size = int(window_length * sample_rate)

        # compute length this way rather than using
        # `segment.length` in case we choose to go the
        # tuple route. Only need this for significance calc
        # TODO: should we allow this as an optional arg that
        # defaults to `None`, in which case we infer length
        # this way. This would extend well to use cases where
        # our analysis period extends to multiple segments,
        # which will probably happen in the short term before
        # we can think about retraining schemes, and still
        # might happen after that depending on how that goes
        length = t[-1] - t[0]

        metrics, times = [], []
        for event_time in event_iter:
            # normalize the time array by the event time
            tc = t - event_time
            mask = tc >= 0
            if (not mask.any()) or mask.all():
                # the event time is either greater or less than
                # all of the GPS times in the segment, so there's
                # nothing we can analyze here
                raise ValueError(
                    "Event time {} doesn't fall within segment "
                    "stretching from {}-{}".format(event_time, t[0], t[-1])
                )

            # find the first index of t such that
            # t[idx] >= event_time, since this is the
            # first timestep of the first kernel that
            # could have contained the event trigger
            idx = mask.argmax()
            event = y[idx : idx + window_size]

            # characterize the isolated event
            if metric == "far":
                characterization = self.far(event)
            elif metric == "significance":
                characterization = self.significance(event, length)

            metrics.append(characterization)
            times.append(tc[idx : idx + window_size])

        # if we only provided a single float event time,
        # return single timeseries for far and t
        if single_event:
            return metrics[0], times[0]

        # otherwise, return an array of these
        # stacked along the 0th dimension
        return np.stack(metrics), np.stack(times)

    def fit(
        self,
        segments: Union[Segment, Iterable[Segment]],
        warm_start: bool = True,
    ) -> None:
        """
        Fit the distribution to the data contained in
        one or more `Segments`.

        Args:
            segments:
                `Segment` or list of `Segments` on which
                to update the distribution
            warm_start:
                Whether to fit the distribution from scratch
                or continue from its existing state.
        """
        if not warm_start:
            self.__post_init__()

        # TODO: accept pathlike and initialize a timeslide?
        if isinstance(segments, Segment):
            segments = [segments]

        for segment in segments:
            y, t = segment.load(self.dataset)
            self.update(y, t)
            self.fnames.extend(segment.fnames)

    def __str__(self):
        return f"{self.__class__.__name__}('{self.dataset}', Tb={self.Tb})"
