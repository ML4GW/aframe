import warnings
from dataclasses import dataclass
from typing import TypeVar

import ligo.segments
import numpy as np

from ledger.injections import InterferometerResponseSet
from ledger.ledger import Ledger, metadata, parameter

SECONDS_PER_YEAR = 31556952  # 60 * 60 * 24 * 365.2425
F = TypeVar("F", np.ndarray, float)


def veto_mask(times: np.ndarray, vetos: np.ndarray) -> np.ndarray:
    """Boolean mask of `times` falling inside any `[start, end)` segment.

    Args:
        times: GPS times to test.
        vetos: `(N, 2)` array of `(start, end)` segment bounds.

    Raises:
        ValueError: if any segment has `start >= end`.
    """
    if len(vetos) == 0:
        return np.zeros(len(times), dtype=bool)

    segs = np.asarray(vetos, dtype=float)
    bad = segs[:, 1] <= segs[:, 0]
    if bad.any():
        raise ValueError(
            f"Veto segments must have start < end, got {segs[bad].tolist()}"
        )

    merged = ligo.segments.segmentlist(
        ligo.segments.segment(start, end) for start, end in segs
    )
    merged.coalesce()
    edges = np.array(merged).ravel()

    # A time being placed at an odd index means that it falls
    # between the start and end time of a veto
    return np.searchsorted(edges, times, side="right") % 2 == 1


@dataclass
class EventSet(Ledger):
    """A set of detected events with associated statistics and timing.

    Attributes:
        detection_statistic: The detection statistic value for each event.
        detection_time: The time of detection for each event.
        shift: List of time shifts defining the timeslide in which
               event was found.
        Tb: Total livetime analyzed in detecting these events, in seconds.
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

    def get_shift(self, shift: np.ndarray) -> "EventSet":
        """Get all events from a specific timeslide shift.

        Args:
            shift: The shift value to filter events by.

        Returns:
            EventSet containing only events from the specified shift.
        """
        # downselect to all events from a given shift
        mask = self.shift == shift
        if self.shift.ndim == 2:
            mask = mask.all(axis=-1)
        return self[mask]

    def nb(self, threshold: F) -> F:
        """Calculate number of events above detection threshold.

        Args:
            threshold: Detection statistic threshold to count above.
                       Can be scalar or array-like.

        Returns:
            Number of events with detection statistic >= threshold.
            If threshold is array-like, returns array of counts.
        """
        if self.is_sorted_by("detection_statistic"):
            return len(self) - np.searchsorted(
                self.detection_statistic, threshold
            )
        warnings.warn(
            "Detection statistic is not sorted. This function "
            "may take a long time for large datasets. To sort, "
            "use the sort_by() function of this object.",
            stacklevel=2,
        )
        try:
            len(threshold)
        except TypeError:
            return (self.detection_statistic >= threshold).sum()
        else:
            stats = self.detection_statistic[:, None]
            return (stats >= threshold).sum(0)

    @property
    def min_far(self):
        """Calculate minimum resolvable false alarm rate (FAR).

        The minimum FAR that can be resolved given the background
        livetime analyzed, in units of yr^-1.

        Returns:
            Minimum FAR in yr^-1.
        """
        return (1 / self.Tb) * SECONDS_PER_YEAR

    def far(self, threshold: F) -> F:
        """Calculate false alarm rate (FAR) for a given detection threshold.

        Computes the FAR in yr^-1 for a given detection threshold. If the
        threshold is above the loudest background event, returns the minimum
        FAR that can be resolved given the accumulated background livetime.

        Args:
            threshold: Detection statistic threshold value.

        Returns:
            FAR in yr^-1. Returns min_far if threshold exceeds all events.
        """
        nb = self.nb(threshold)
        far = SECONDS_PER_YEAR * nb / self.Tb
        return np.maximum(far, self.min_far)

    def significance(self, threshold: F, T: float) -> F:
        """Calculate significance of detection at given threshold.

        Represents the likelihood that at least one event with detection
        statistic value >= threshold will occur after observing this
        distribution for a period T. See https://arxiv.org/pdf/1508.02357.pdf
        equation 17 for theoretical details.

        Args:
            threshold: The detection statistic threshold to compare against.
            T: The length of the analysis period in which the detection
               statistic was measured, in seconds.

        Returns:
            Significance value (probability between 0 and 1).
        """

        nb = self.nb(threshold)
        return 1 - np.exp(-T * (1 + nb) / self.Tb)

    def threshold_at_far(self, far: float):
        """Find detection statistic threshold corresponding to a FAR value.

        Args:
            far: Target FAR in Hz.

        Returns:
            Detection statistic threshold corresponding to the given FAR.
        """
        livetime = self.Tb
        num_events = livetime * far
        if self.is_sorted_by("detection_statistic"):
            return self.detection_statistic[-int(num_events)]
        warnings.warn(
            "Detection statistic is not sorted. This function "
            "may take a long time for large datasets. To sort, "
            "use the sort_by() function of this object.",
            stacklevel=2,
        )
        det_stats = np.sort(self.detection_statistic)
        return det_stats[-int(num_events)]

    def apply_vetos(
        self,
        vetos: list[tuple[float, float]],
        idx: int,
        return_mask: bool = False,
    ):
        """Apply time-based vetoes to remove events.

        Args:
            vetos: List of (start_time, end_time) tuples defining veto periods.
            idx: Index of the shift/interferometer to apply vetoes for.
            return_mask: If True, return both filtered events and veto mask.
                    Defaults to False.

        Returns:
            If return_mask is False: Vetoed EventSet.
            If return_mask is True: Tuple of (vetoed_events, veto_mask).
        """
        shift = self.shift[:, idx]
        times = self.detection_time + shift
        mask = veto_mask(times, vetos)
        result = self[~mask]
        if return_mask:
            return result, mask
        return result


@dataclass
class RecoveredInjectionSet(EventSet, InterferometerResponseSet):
    """A set of injected signals recovered as detected events.

    Combines detected event information with injected waveform parameters,
    storing data about detected events that matched injected signals.
    """

    @classmethod
    def compare_metadata(cls, key, ours, theirs):
        if key == "num_injections":
            return InterferometerResponseSet.compare_metadata(
                key, ours, theirs
            )
        return super().compare_metadata(key, ours, theirs)

    @classmethod
    def recover(cls, events: EventSet, injections: InterferometerResponseSet):
        """Match detected events to injected signals using injection time.

        For each injection, finds the event closest in time at the same
        timeslide shift.

        Args:
            events: EventSet containing detected events.
            injections: InterferometerResponseSet containing injected signals.

        Returns:
            RecoveredInjectionSet with matched event and injection data.
        """
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
                **kwargs,
            )
            obj.append(subobj)

        obj.Tb = events.Tb
        return obj
