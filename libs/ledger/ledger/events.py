from dataclasses import dataclass
from typing import List, Tuple, TypeVar

import numpy as np
from ledger.injections import InterferometerResponseSet
from ledger.ledger import Ledger, metadata, parameter

SECONDS_IN_YEAR = 31556952
F = TypeVar("F", np.ndarray, float)


@dataclass
class EventSet(Ledger):
    detection_statistic: np.ndarray = parameter()
    gps_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()
    Tb: float = metadata(default=0)

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "Tb":
            return ours + theirs
        return Ledger.compare_metadata(key, ours, theirs)

    def get_shift(self, shift):
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def nb(self, threshold: F) -> F:
        try:
            len(threshold)
        except TypeError:
            return (self.detection_statistic >= threshold).sum()
        else:
            stats = self.detection_statistic[:, None]
            return (stats >= threshold).sum(0)

    def far(self, threshold: F) -> F:
        nb = self.nb(threshold)
        return SECONDS_IN_YEAR * nb / self.Tb

    def significance(self, threshold: F, T: float) -> F:
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

    def apply_vetos(self, vetos: np.ndarray):
        # TimeSlideEventSet does not have shift information,
        # so apply vetoes as if the shift is zero
        mask = np.logical_and(
            vetos[:, :1] < self.time, vetos[:, 1:] > self.time
        )

        # mark a background event as vetoed
        # if it falls into _any_ of the segments
        veto_mask = mask.any(axis=0)

        # TODO: have an 'inplace=False' option that returns a new object?
        return self[~veto_mask]


# inherit from TimeSlideEventSet since injection
# will already have shift information recorded
@dataclass
class RecoveredInjectionSet(InterferometerResponseSet):
    detection_statistic: np.ndarray = parameter()
    detection_time: np.ndarray = parameter()

    @classmethod
    def recover(cls, events: EventSet, injections: InterferometerResponseSet):
        obj = cls()
        for shift in np.unique(events.shift, axis=-1):
            # get the all events and injections at the current shift
            evs = events.get_shift(shift)
            injs = injections.get_shift(shift)

            # for each injection, find the event closest to it in time
            # TODO: should this just look _after_ the event?
            diffs = np.abs(injs.gps_time[:, None] - evs.gps_time)
            idx = diffs.argmin(axis=-1)
            evs = evs[idx]

            # create a RecoveredInjection object for just this
            # shift and then append it onto our running ledger
            fields = set(cls.__dataclass_fields__)
            fields &= set(injs.__dataclass_fields__)
            kwargs = {k: getattr(injs, k) for k in fields}
            subobj = cls(
                detection_statistic=evs.detection_statistic,
                detection_time=evs.gps_time,
                **kwargs
            )
            obj.append(subobj)
        return obj

    def apply_vetos(self, vetos: List[Tuple[float, float]], idx: int):
        # idx corresponds to the index of the shift
        # (i.e. which ifo to apply vetoes for)
        shifts = self.shift[:, idx]
        times = self.time + shifts

        mask = np.logical_and(vetos[:, :1] < times, vetos[:, 1:] > times)

        # mark a background event as vetoed
        # if it falls into _any_ of the segments
        veto_mask = mask.any(axis=0)

        # TODO: Should we adjust the num_injections parameter?
        # or maybe add a num_vetoed parameter?
        return self[~veto_mask]
