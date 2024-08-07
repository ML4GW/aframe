import warnings
from typing import Callable, Optional

import astropy
import astropy.cosmology
import numpy as np
from astropy.cosmology import Planck15
from ledger.events import SECONDS_IN_YEAR, EventSet, F, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.stats import gaussian_kde


def fit_background_model(
    background: EventSet,
    split: Optional[float] = None,
) -> Callable:
    """
    Fit a model to the background detection statistic distribution,
    scaled so that the model output can be interpreted as the
    expected rate density of background events

    Args:
        background:
            EventSet object corresponding to background events
            recovered in a search over timeslides
        split:
            The detection statistic at which to switch from using
            a KDE to fit the background to using an exponential
            fit to the background. If None, the split point is
            estimated as the point at which the PDF of the KDE
            drops below 1/sqrt(N), where N is the number of
            background events

    Returns:
        A callable that takes a detection statistic and returns the
        expected rate density of background events at that statistic
    """
    kde = gaussian_kde(background.detection_statistic)

    # Estimate the peak of the distribution
    samples = np.linspace(
        background.detection_statistic.min(),
        background.detection_statistic.max(),
        100,
    )
    pdf = kde(samples)
    peak_idx = np.argmax(pdf)

    if split is not None:
        stop = np.argmin(np.abs(samples - split))
        threshold_pdf_value = pdf[stop]
        start = np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
    else:
        # Determine the range of values to use for fitting
        # a line to a portion of the pdf.
        # Roughly, we have too few samples to properly
        # estimate the KDE once the pdf drops below 1/sqrt(N)
        threshold_pdf_value = 1 / np.sqrt(len(background))
        start = np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
        stop = np.argmin(pdf[peak_idx:] > threshold_pdf_value) + peak_idx

    # Fit a line to the log pdf of the region prior to the split point
    fit_samples = samples[start:stop]
    fit = Polynomial.fit(fit_samples, np.log(pdf[start:stop]), 1)
    threshold_statistic = samples[stop]

    if not np.isclose(
        kde(threshold_statistic), fit(threshold_statistic), rol=1e-2
    ):
        warnings.warn(
            "The KDE and exponential fit have a greater than 1% discrepancy "
            "at the split point. It is recommended to examine the model to "
            "determine the cause."
        )

    scale_factor = len(background) * SECONDS_IN_YEAR / background.Tb

    def background_model(stats: F) -> F:
        """
        Args:
            stats:
                Detection statistics for which to compute the background model
        """
        return (
            np.piecewise(
                stats,
                [stats < threshold_statistic, stats >= threshold_statistic],
                [kde, lambda x: np.exp(fit(x))],
            )
            * scale_factor
        )

    return background_model


def _volume_element(cosmology, z):
    return cosmology.differential_comoving_volume(z).value / (1 + z)


def get_injected_volume(
    foreground: RecoveredInjectionSet,
    rejected: InjectionParameterSet,
    cosmology=Planck15,
) -> float:
    """
    Calculate the volume of the universe in which injections were made.

    Args:
        foreground:
            RecoveredInjectionSet object corresponding to an
            injection campaign
        rejected:
            InjectionParameterSet object corresponding to signals
            that were simulated but rejected due to low SNR
        cosmology:
            The cosmology to use when calculating the injected volume

    Returns:
        The injection volume in cubic gigaparsecs
    """
    zmin = min(rejected.redshift.min(), foreground.redshift.min())
    zmax = max(rejected.redshift.max(), foreground.redshift.max())
    dec_min = min(rejected.dec.min(), foreground.dec.min())
    dec_max = max(rejected.dec.max(), foreground.dec.max())

    volume, _ = quad(lambda z: _volume_element(cosmology, z), zmin, zmax)
    theta_max = np.pi / 2 - dec_min
    theta_min = np.pi / 2 - dec_max
    omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
    return volume * omega / 1e9


def fit_foreground_model(
    foreground: RecoveredInjectionSet,
    rejected: InjectionParameterSet,
    astro_event_rate: float,
    cosmology: astropy.cosmology.Cosmology = Planck15,
) -> Callable:
    """
    Fit a model to the foreground detection statistic distribution,
    scaled so that the model output can be interpreted as the
    expected rate density of events

    Args:
        foreground:
            RecoveredInjectionSet object corresponding to an
            injection campaign
        rejected:
            InjectionParameterSet object corresponding to signals
            that were simulated but rejected due to low SNR
        astro_event_rate:
            The rate density of events for the relevent population.
            Expected units are events per year per cubic gigaparsec
        cosmology:
            The cosmology to use when calculating the injected volume

    Returns:
        A callable that takes a detection statistic and returns the
        expected rate density of events at that statistic
    """
    injected_volume = get_injected_volume(foreground, rejected, cosmology)
    total_injections = len(foreground) + len(rejected)
    scale_factor = (
        astro_event_rate * injected_volume * len(foreground) / total_injections
    )
    kde = gaussian_kde(foreground.detection_statistic)

    def foreground_model(stats: F) -> F:
        """
        Args:
            stats:
                Detection statistics for which to compute the foreground model
        """
        return kde(stats) * scale_factor

    return foreground_model


def fit_p_astro(
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected: InjectionParameterSet,
    astro_event_rate: float,
    cosmology: astropy.cosmology.Cosmology = Planck15,
) -> Callable:
    """
    Compute p_astro as the ratio of the expected foreground rate
    to the sum of the expected foreground rate and the expected
    background event rate for a given set of detection
    statistics
    Args:
        background:
            EventSet object corresponding to background events
            recovered in a search over timeslides
        foreground:
            RecoveredInjectionSet object corresponding to an
            injection campaign
        rejected:
            InjectionParameterSet object corresponding to signals
            that were simulated but rejected due to low SNR
        astro_event_rate:
            The rate density of events for the relevent population.
            Expected units are events per year per cubic gigaparsec
        cosmology:
            The cosmology to use when calculating the injected volume
    """
    background_model = fit_background_model(background=background)
    foreground_model = fit_foreground_model(
        foreground=foreground,
        rejected=rejected,
        astro_event_rate=astro_event_rate,
        cosmology=cosmology,
    )

    def p_astro(stats: F) -> F:
        """
        Args:
            stats:
                Detection statistics for which to compute p_astro
        """
        background_rate = background_model(stats)
        foreground_rate = foreground_model(stats)
        return foreground_rate / (foreground_rate + background_rate)

    return p_astro
