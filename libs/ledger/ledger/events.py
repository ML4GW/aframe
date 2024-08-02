import copy
from dataclasses import dataclass
from typing import List, Tuple, TypeVar

import numpy as np
from ledger.injections import InterferometerResponseSet
from ledger.ledger import Ledger, metadata, parameter

SECONDS_IN_YEAR = 31556952
F = TypeVar("F", np.ndarray, float)


@dataclass
class EventSet(Ledger):
    """
    A set of detected events

    Args:
        detection_statistic:
            The detection statistic for each event
        detection_time:
            The time of each event
        shift:
            List of time shifts corresponding
            to timeslide in which event was found
        Tb:
            The total livetime analyzed in detecting these events
    """

    detection_statistic: np.ndarray = parameter()
    detection_time: np.ndarray = parameter()
    shift: np.ndarray = parameter()
    Tb: float = metadata(default=0)

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        # accumulate background time when merging or appending
        if key == "Tb":
            return ours + theirs
        return super().compare_metadata(key, ours, theirs)
    
    @property
    def min_far(self):
        """
        Lowest FAR in Hz that can be resolved given background
        livetime analyzed
        """
        return 1 / self.Tb * SECONDS_IN_YEAR
    
    def far(self, threshold: F) -> F:
        """
        Far in Hz for a given detection statistic threshold, or
        the minimum FAR that can be resolved
        given the accumulated background livetime
        """
        nb = self.nb(threshold)
        return SECONDS_IN_YEAR * nb / self.Tb

    def get_shift(self, shift: np.ndarray) -> "EventSet":
        # downselect to all events from a given shift
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def nb(self, threshold: F) -> F:
        """
        The number of events with a detection statistic
        greater than or equal to `threshold`
        """
        try:
            len(threshold)
        except TypeError:
            return (self.detection_statistic >= threshold).sum()
        else:
            stats = self.detection_statistic[:, None]
            return (stats >= threshold).sum(0)

    def far(self, threshold: F) -> F:
        """
        The false alarm rate for a given threshold
        """
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
    
    def threshold_at_far(self, far: float):
        """
        Return the detection statistic threshold
        that corresponds to a given far in Hz
        """
        livetime = self.Tb
        num_events = livetime * far
        det_stats = sorted(self.detection_statistic)
        return det_stats[-int(num_events)]

    def apply_vetos(
        self,
        vetos: List[Tuple[float, float]],
        idx: int,
        chunk_size: int = 500000,
        inplace: bool = False,
        return_mask: bool = False,
    ):
        # idx corresponds to the index of the shift
        # (i.e., which ifo to apply vetoes for)
        shifts = self.shift[:, idx]
        times = self.detection_time + shifts

        # array of False, no vetoes applied yet
        veto_mask = np.zeros(len(times), dtype=bool)

        # process triggers in chunks to avoid memory issues
        for i in range(0, len(times), chunk_size):
            # apply vetos for this chunk of times
            chunk_times = times[i : i + chunk_size]
            mask = np.logical_and(
                vetos[:, :1] < chunk_times, vetos[:, 1:] > chunk_times
            )
            veto_mask[i : i + chunk_size] = mask.any(axis=0)

        if inplace:
            result = self[~veto_mask]
        else:
            result = copy.deepcopy(self)[~veto_mask]

        if return_mask:
            return result, veto_mask
        return result


@dataclass
class RecoveredInjectionSet(EventSet, InterferometerResponseSet):
    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "num_injections":
            return InterferometerResponseSet.compare_metadata(
                key, ours, theirs
            )
        return super().compare_metadata(key, ours, theirs)

    @classmethod
    def recover(cls, events: EventSet, injections: InterferometerResponseSet):
        obj = cls()
        for shift in np.unique(events.shift, axis=0):
            # get the all events and injections at the current shift
            evs = events.get_shift(shift)
            injs = injections.get_shift(shift)

            # for each injection, find the event closest to it in time
            # TODO: should this just look _after_ the event?
            diffs = np.abs(injs.injection_time[:, None] - evs.detection_time)
            idx = diffs.argmin(axis=-1)
            evs = evs[idx]

            # create a RecoveredInjection object for just this
            # shift and then append it onto our running ledger
            fields = set(cls.__dataclass_fields__)
            fields &= set(injs.__dataclass_fields__)

            kwargs = {k: getattr(injs, k) for k in fields}
            kwargs["num_injections"] = len(injs)

            subobj = cls(
                detection_statistic=evs.detection_statistic,
                detection_time=evs.detection_time,
                **kwargs
            )
            obj.append(subobj)

        obj.Tb = events.Tb
        return obj
