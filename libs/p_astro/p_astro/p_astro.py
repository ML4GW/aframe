from typing import Callable

import astropy
import astropy.cosmology
import numpy as np
from astropy.cosmology import Planck15
from ledger.events import SECONDS_IN_YEAR, EventSet, F, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.stats import gaussian_kde


def fit_noise_model(background: EventSet) -> Callable:
    # perform kde on bulk
    kde = gaussian_kde(background.detection_statistic)

    # Estimate the peak of the distribution
    samples = np.linspace(
        min(background.detection_statistic),
        max(background.detection_statistic),
        100,
    )
    pdf = kde(samples)

    # Determine the range of values to use for fitting
    # a line to a portion of the pdf.
    # Roughly, we have too few samples to properly
    # estimate the KDE once the pdf drops below 1/sqrt(N)
    peak_idx = np.argmax(pdf)
    threshold_pdf_value = 1 / np.sqrt(len(background.detection_statistic))
    start = np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
    stop = np.argmin(pdf[peak_idx:] > threshold_pdf_value) + peak_idx

    # Fit a line to the log pdf of the region
    fit_samples = samples[start:stop]
    fit = Polynomial.fit(fit_samples, np.log(pdf[start:stop]), 1)
    threshold_statistic = samples[start]

    scale_factor = len(background) * SECONDS_IN_YEAR / background.Tb

    def noise_model(stats: F) -> F:
        return (
            np.piecewise(
                stats,
                [stats < threshold_statistic, stats >= threshold_statistic],
                [kde, np.exp(fit(stats))],
            )
            * scale_factor
        )

    return noise_model


def _volume_element(cosmology, z):
    return cosmology.differential_comoving_volume(z).value / (1 + z)


def get_injected_volume(
    zmin: float,
    zmax: float,
    dec_min: float,
    dec_max: float,
    cosmology=Planck15,
) -> float:
    volume, _ = quad(lambda z: _volume_element(cosmology, z), zmin, zmax)
    theta_max = np.pi / 2 - dec_min
    theta_min = np.pi / 2 - dec_max
    omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
    return volume * omega / 1e9


def fit_signal_model(
    rejected: InjectionParameterSet,
    foreground: RecoveredInjectionSet,
    astro_event_rate: float,
    cosmology: astropy.cosmology.Cosmology = Planck15,
) -> Callable:
    zmin = min(rejected.redshift.min(), foreground.redshift.min())
    zmax = max(rejected.redshift.max(), foreground.redshift.max())
    dec_min = min(rejected.dec.min(), foreground.dec.min())
    dec_max = max(rejected.dec.max(), foreground.dec.max())

    injected_volume = get_injected_volume(
        zmin, zmax, dec_min, dec_max, cosmology
    )
    total_injections = len(foreground) + len(rejected)

    scale_factor = (
        astro_event_rate * injected_volume * len(foreground) / total_injections
    )

    kde = gaussian_kde(foreground.detection_statistic)

    def signal_model(stats: F) -> F:
        return kde(stats) * scale_factor

    return signal_model


# some helper function for doing the whole fitting from our ledger objects
def fit_p_astro(
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected: InjectionParameterSet,
    astro_event_rate: float,
    cosmology: astropy.cosmology.Cosmology = Planck15,
) -> Callable:
    noise_model = fit_noise_model(background=background)
    signal_model = fit_signal_model(
        rejected=rejected,
        foreground=foreground,
        astro_event_rate=astro_event_rate,
        cosmology=cosmology,
    )

    def p_astro(stats: F, min_det_stat: float) -> F:
        p_astro = np.zeros_like(stats)
        mask = stats > min_det_stat
        stats = stats[mask]
        noise_rate = noise_model(stats)
        foreground_rate = signal_model(stats)
        p_astro[mask] = foreground_rate / (foreground_rate + noise_rate)
        return p_astro

    return p_astro
