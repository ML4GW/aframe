from typing import List

import numpy as np
from ledger.events import SECONDS_IN_YEAR, EventSet, F, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from numpy.polynomial import Polynomial
from scipy.stats import gaussian_kde


class NoiseModel:
    def __init__(
        self,
        background_statistics: List[float],
        Tb: float,
    ):
        # Define the KDE using all statisics and the default bandwidth
        # TODO: evaluate whether there's any benefit to optimizing bandwidth
        self.background_statistics = background_statistics
        self.Tb = Tb
        self.kde = gaussian_kde(self.background_statistics)

        # Estimate the peak of the distribution
        samples = np.linspace(
            min(self.background_statistics),
            max(self.background_statistics),
            100,
        )
        pdf = self.kde(samples)

        # Determine the range of values to use for fitting
        # a line to a portion of the pdf.
        # Roughly, we have too few samples to properly
        # estimate the KDE once the pdf drops below 1/sqrt(N)
        peak_idx = np.argmax(pdf)
        threshold_pdf_value = 1 / np.sqrt(len(self.background_statistics))
        start = np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
        stop = np.argmin(pdf[peak_idx:] > threshold_pdf_value) + peak_idx

        # Fit a line to the log pdf of the region
        fit_samples = samples[start:stop]
        self.background_fit = Polynomial.fit(
            fit_samples, np.log(pdf[start:stop]), 1
        )
        self.threshold_statistic = samples[start]

    def __call__(self, statistics: F) -> F:
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

        return background_density * len(self.background_statistics) / self.Tb


class SignalModel:
    """
    This is mostly just a wrapper around gaussian_kde,
    but we can make this more sophisticated someday
    """

    def __init__(
        self,
        foreground_statistics: List[float],
        total_injections: int,
        injected_volume: float,
        astro_event_rate: float,
    ):
        # TODO: evaluate whether there's any benefit to optimizing bandwidth
        self.foreground_kde = gaussian_kde(foreground_statistics)
        self.scaling_factor = (
            astro_event_rate
            * injected_volume
            * len(foreground_statistics)
            / total_injections
        )

    def __call__(self, statistics: F) -> F:
        """
        Calculate the expected signal event rate for a given
        statistic or set of statistics
        """
        return self.foreground_kde(statistics) * self.scaling_factor


def compute_p_astro(
    detection_statistic: F,
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected_params: InjectionParameterSet,
    injected_volume: float,
    astro_event_rate: float,
    min_det_stat: float = -np.inf,
) -> F:
    """
    Compute p_astro as the ratio of the expected signal rate
    to the sum of the expected signal rate and the expected
    background event rate for a given set of detection
    statistics

    Args:
        detection_statistic:
            A value or set of values for which to calculate the
            corresponding p_astro
        background:
            EventSet object corresponding to a background model
        foreground:
            RecoveredInjectionSet object corresponding to an
            injection campaign
        rejected_params:
            InjectionParameterSet object corresponding to signals
            that were simulated but rejected due to low SNR
        injected_volume:
            The astrophysical volume into which injections were
            made during the injection campaign
        astro_event_rate:
            The rate density of events for the relavent population.
            Should have the same spatial units as injected_volume
        min_det_stat:
            Then minimum detection statistic for which to compute
            p_astro. Detection statistics below this value will
            be assigned a p_astro of 0
    """
    p_astro = np.zeros_like(detection_statistic)
    mask = detection_statistic >= min_det_stat
    detection_statistic = detection_statistic[mask]

    total_injections = len(foreground) + len(rejected_params)

    signal_model = SignalModel(
        foreground.detection_statistic,
        total_injections,
        injected_volume,
        astro_event_rate,
    )

    Tb = background.Tb / SECONDS_IN_YEAR
    noise_model = NoiseModel(background.detection_statistic, Tb)

    foreground_rate = signal_model(detection_statistic)
    background_rate = noise_model(detection_statistic)

    p_astro[mask] += foreground_rate / (foreground_rate + background_rate)
    return p_astro
