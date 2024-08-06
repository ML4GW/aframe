import copy
from dataclasses import dataclass
from typing import List, Tuple, TypeVar

import numpy as np
from astropy.cosmology import Planck15
from ledger.injections import InjectionParameterSet, InterferometerResponseSet
from ledger.ledger import Ledger, metadata, parameter
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.stats import gaussian_kde

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
        far = SECONDS_IN_YEAR * nb / self.Tb
        return np.maximum(far, self.min_far)

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
        det_stats = np.sort(self.detection_statistic)
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

    def fit_noise_model(self):
        self.kde = gaussian_kde(self.detection_statistic)

        # Estimate the peak of the distribution
        samples = np.linspace(
            min(self.detection_statistic),
            max(self.detection_statistic),
            100,
        )
        pdf = self.kde(samples)

        # Determine the range of values to use for fitting
        # a line to a portion of the pdf.
        # Roughly, we have too few samples to properly
        # estimate the KDE once the pdf drops below 1/sqrt(N)
        peak_idx = np.argmax(pdf)
        threshold_pdf_value = 1 / np.sqrt(len(self.detection_statistic))
        start = np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
        stop = np.argmin(pdf[peak_idx:] > threshold_pdf_value) + peak_idx

        # Fit a line to the log pdf of the region
        fit_samples = samples[start:stop]
        self.background_fit = Polynomial.fit(
            fit_samples, np.log(pdf[start:stop]), 1
        )
        self.threshold_statistic = samples[start]

    def noise_model(self, statistics: F) -> F:
        """
        Calculate the expected background event rate for a given
        statistic or set of statistics
        """
        try:
            len(statistics)
        except TypeError:
            if statistics < self.threshold_statistic:
                background_density = self.kde(statistics)
            else:
                background_density = np.exp(self.background_fit(statistics))
        else:
            background_density = np.zeros_like(statistics)
            mask = statistics < self.threshold_statistic

            background_density[mask] = self.kde(statistics[mask])
            background_density[~mask] = np.exp(
                self.background_fit(statistics[~mask])
            )

        return background_density * len(self.detection_statistic) / self.Tb

    # TODO: This doesn't really feel like it goes here.
    # Is there a better place?
    def p_astro(
        self,
        background: "EventSet",
        foreground: "RecoveredInjectionSet",
        rejected_params: InjectionParameterSet,
        astro_event_rate: float,
        statistics: F = None,
        cosmology=Planck15,
        min_det_stat: float = -np.inf,
    ) -> F:
        """
        Compute p_astro as the ratio of the expected signal rate
        to the sum of the expected signal rate and the expected
        background event rate for a given set of detection
        statistics

        Args:
            background:
                EventSet object corresponding to a background model
            foreground:
                RecoveredInjectionSet object corresponding to an
                injection campaign
            rejected_params:
                InjectionParameterSet object corresponding to signals
                that were simulated but rejected due to low SNR
            astro_event_rate:
                The rate density of events for the relavent population.
                Should have the same spatial units as injected_volume
            cosmology:
                The cosmology to use when calculating the injected volume
            min_det_stat:
                Then minimum detection statistic for which to compute
                p_astro. Detection statistics below this value will
                be assigned a p_astro of 0
        """
        if statistics is None:
            statistics = self.detection_statistic
        p_astro = np.zeros_like(statistics)
        mask = statistics >= min_det_stat
        statistics = statistics[mask]

        foreground.fit_signal_model(
            rejected_params, astro_event_rate, cosmology
        )
        foreground_rate = foreground.signal_model(statistics)
        background.fit_noise_model()
        background_rate = background.noise_model(statistics)

        p_astro[mask] += foreground_rate / (foreground_rate + background_rate)
        return p_astro


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

    def fit_signal_model(
        self,
        rejected_params: InjectionParameterSet,
        astro_event_rate: float,
        cosmology=Planck15,
    ):
        """
        Args:
            rejected_params:
                InjectionParameterSet object corresponding to signals
                that were simulated but rejected due to low SNR
            astro_event_rate:
                The rate density of events for the relavent population.
                Should have the same spatial units as injected_volume
            cosmology:
                The cosmology to use when calculating the injected volume
        """
        self.scaling_factor = (
            astro_event_rate
            * self.injected_volume(cosmology)
            * len(self)
            / (len(self) + len(rejected_params))
        )
        self.kde = gaussian_kde(self.detection_statistic)

    def signal_model(
        self,
        statistics: F,
    ) -> F:
        return self.kde(statistics) * self.scaling_factor

    def _volume_element(self, cosmology, z):
        return cosmology.differential_comoving_volume(z).value / (1 + z)

    def injected_volume(self, cosmology=Planck15):
        zmin, zmax = self.redshift.min(), self.redshift.max()
        decmin, decmax = self.dec.min(), self.dec.max()

        volume, _ = quad(
            lambda z: self._volume_element(cosmology, z), zmin, zmax
        )
        theta_max = np.pi / 2 - decmin
        theta_min = np.pi / 2 - decmax
        omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
        return volume * omega
