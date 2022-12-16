from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from bbhnet.io.timeslides import Segment

SECONDS_IN_YEAR = 31556952

SEGMENT_LIKE = Union[Segment, Iterable[Segment], Tuple[np.ndarray, np.ndarray]]


@dataclass
class Distribution:
    """A base class for a Distribution of events

    Args:
        dataset:
            The name of the dataset used to generate events. Used to load data
            when fitting directly from Segment objects
        ifos:
            The ifos from which these events are derived.

    """

    dataset: str
    ifos: Iterable[str]

    def __post_init__(self):
        self.Tb = 0
        self.events = np.array([])
        self.event_times = np.array([])
        self.shifts = np.empty((0, len(self.ifos)))

    def write(self, path: Path):
        raise NotImplementedError

    def update(self, x: np.ndarray, t: np.ndarray, shifts: np.ndarray):
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

    def fit(
        self,
        segments: SEGMENT_LIKE,
        shifts: Optional[np.ndarray] = None,
        warm_start: bool = True,
    ) -> None:
        """
        Fit the distribution to the data contained in
        one or more `Segments`.

        Args:
            segments:
                `Segment`, list of `Segments`, or Tuple (y, t) of np.ndarrays
                on which to update the distribution
            shifts:
                If passing a Tuple as segments, the time shifts
                of interferometers used to generate the output.
                The shifts should correspond one-to-one with the
                ifos attribute of the Distribution.
            warm_start:
                Whether to fit the distribution from scratch
                or continue from its existing state.
        """
        if not warm_start:
            self.__post_init__()

        if isinstance(segments, Tuple):
            self.update(*segments, shifts)
            return

        if isinstance(segments, Segment):
            segments = [segments]

        for segment in segments:
            shifts = segment.shifts
            y, t = segment.load(self.dataset)
            self.update(y, t, shifts)

    def apply_vetoes(self, **vetoes: np.ndarray):
        """Remove events if the time in any of the interferometers lies in
        a vetoed segment.

        Args:
            vetoes:
                np.ndarray of shape (n_segments, 2) corresponding to segments
                that should be vetoed.

        """

        for ifo, vetoes in vetoes.items():

            # find shifts corresponding to this ifo
            # and calculate event times for this ifo
            try:
                shift_arg = self.ifos.index(ifo)
            except ValueError:
                raise ValueError(
                    f"Attempting to apply vetoes to ifo {ifo},"
                    f"but {ifo} is not an ifo in this distribution"
                )
            times = self.event_times - self.shifts[:, shift_arg]

            # determine event times that are in vetoed segments
            mask = np.ones(len(times), dtype=bool)
            for t0, tf in vetoes:
                mask &= (t0 >= times) | (times >= tf)

            # apply mask
            event_times = self.event_times[mask]
            shifts = self.shifts[mask]
            events = self.events[mask]

            self.event_times = event_times
            self.shifts = shifts
            self.events = events

    def __str__(self):
        return f"{self.__class__.__name__}('{self.dataset}', Tb={self.Tb})"
