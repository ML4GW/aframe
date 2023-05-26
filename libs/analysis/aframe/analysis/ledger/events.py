from dataclasses import dataclass
from typing import List, TypeVar, Union

import numpy as np

from aframe.analysis.ledger.injections import InterferometerResponseSet
from aframe.analysis.ledger.ledger import Ledger, metadata, parameter

SECONDS_IN_YEAR = 31556952
F = TypeVar("F", np.ndarray, float)


@dataclass
class TimeSlideEventSet(Ledger):
    detection_statistic: np.ndarray = parameter()
    time: np.ndarray = parameter()
    Tb: float = metadata(default=0)

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "Tb":
            return ours + theirs
        return Ledger.compare_metadata(key, ours, theirs)

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


@dataclass
class EventSet(TimeSlideEventSet):
    shift: np.ndarray = parameter()

    def get_shift(self, shift):
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    @classmethod
    def from_timeslide(cls, event_set: TimeSlideEventSet, shift: List[float]):
        shifts = np.array([shift] * len(event_set))
        d = {k: getattr(event_set, k) for k in event_set.__dataclass_fields__}
        return cls(shift=shifts, **d)


# inherit from TimeSlideEventSet since injection
# will already have shift information recorded
@dataclass
class RecoveredInjectionSet(TimeSlideEventSet, InterferometerResponseSet):
    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "num_injections":
            return InterferometerResponseSet.compare_metadata(
                key, ours, theirs
            )
        return EventSet.compare_metadata(key, ours, theirs)

    @staticmethod
    def get_idx_for_shift(
        event_times: np.ndarray, injection_times: np.ndarray
    ) -> np.ndarray:
        diffs = np.abs(injection_times[:, None] - event_times)
        return diffs.argmin(axis=-1)

    @classmethod
    def join(
        cls, events: TimeSlideEventSet, responses: InterferometerResponseSet
    ):
        kwargs = {}
        for obj in [events, responses]:
            for key in obj.__dataclass_fields__:
                if key in cls.__dataclass_fields__:
                    value = getattr(obj, key)
                    kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def recover(
        cls,
        events: Union[TimeSlideEventSet, EventSet],
        responses: InterferometerResponseSet,
    ):
        if isinstance(events, EventSet):
            obj = cls()
            for shift in events.shift.unique(axis=-1):
                shift_events = events.get_shift(shift)
                shift_responses = responses.get_shift(shift)
                idx = cls.get_idx_for_shift(
                    shift_events.time, shift_responses.gps_time
                )
                shift_events = shift_events[idx]
                subobj = cls.join(shift_events, shift_responses)
                obj.append(subobj)
            obj.Tb = events.Tb
            return obj

        idx = cls.get_idx_for_shift(events.time, responses.gps_time)
        return cls.join(events[idx], responses)
