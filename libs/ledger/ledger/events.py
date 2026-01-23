import copy
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import TypeVar

import numpy as np
from tqdm import tqdm

from ledger.injections import InterferometerResponseSet
from ledger.ledger import Ledger, metadata, parameter

SECONDS_IN_YEAR = 31556952
F = TypeVar("F", np.ndarray, float)


def process_chunk(args):
    """Process a chunk of events to apply veto masks.

    Args:
        args: Tuple of (chunk_times, vetos, chunk_index) where:
              - chunk_times: Time values to check against vetoes
              - vetos: Array of (start, end) time pairs
              - chunk_index: Index of the current chunk

    Returns:
        Tuple of (chunk_index, veto_mask) with boolean mask
        indicating which events are vetoed.
    """
    chunk_times, vetos, i = args
    mask = np.logical_and(
        vetos[:, :1] < chunk_times, vetos[:, 1:] > chunk_times
    )
    return i, mask.any(axis=0)


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
        return (1 / self.Tb) * SECONDS_IN_YEAR

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
        far = SECONDS_IN_YEAR * nb / self.Tb
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
        chunk_size: int = 500000,
        inplace: bool = False,
        return_mask: bool = False,
    ):
        """Apply time-based vetoes to remove events.

        Removes events that fall within specified time intervals using
        multiprocessing for efficiency on large datasets.

        Args:
            vetos: List of (start_time, end_time) tuples defining veto periods.
            idx: Index of the shift/interferometer to apply vetoes for.
            chunk_size: Number of events to process per chunk.
                Defaults to 500000.
            inplace: If True, modify this object. If False, return copy.
                    Defaults to False.
            return_mask: If True, return both filtered events and veto mask.
                    Defaults to False.

        Returns:
            If return_mask is False: Vetoed EventSet.
            If return_mask is True: Tuple of (vetoed_events, veto_mask).
        """
        # idx corresponds to the index of the shift
        # (i.e., which ifo to apply vetoes for)
        shifts = self.shift[:, idx]
        times = self.detection_time + shifts

        # array of False, no vetoes applied yet
        veto_mask = np.zeros(len(times), dtype=bool)

        # split times into chunks;
        # keep track of the index of the chunk
        # for mp purposes so we can unpack the results later
        chunks = [
            (times[idx : idx + chunk_size], vetos, i)
            for i, idx in enumerate(range(0, len(times), chunk_size))
        ]

        num_cpus = min(cpu_count(), len(chunks))
        with Pool(num_cpus) as pool:
            results = pool.imap_unordered(process_chunk, chunks)

            # combine results
            with tqdm(total=len(chunks)) as pbar:
                for i, result in results:
                    veto_mask[i * chunk_size : (i + 1) * chunk_size] = result
                    pbar.update()

        if inplace:
            result = self[~veto_mask]
        else:
            result = copy.deepcopy(self)[~veto_mask]

        if return_mask:
            return result, veto_mask
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
