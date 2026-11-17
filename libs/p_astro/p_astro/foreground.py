from typing import TYPE_CHECKING

import numpy as np
from astropy.cosmology import Cosmology
from ledger.events import RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from scipy.stats import gaussian_kde
from utils.cosmology import (
    DEFAULT_COSMOLOGY,
    get_astrophysical_volume,
    log_dNdz,
)

if TYPE_CHECKING:
    from bilby.core.prior import PriorDict


class ForegroundModel:
    """
    Base class for foreground models.

    Subclasses should implement the `fit` and `__call__` methods.

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
        source_prior:
            The instantiated prior the injections were drawn from.
            Its redshift prior sets the weights that reweight the
            injections to the target redshift distribution
            (uniform in comoving volume and source-frame time), and
            its redshift and declination bounds set the injected volume.
        cosmology:
            The cosmology to use when calculating the injected volume
    """

    def __init__(
        self,
        foreground: RecoveredInjectionSet,
        rejected: InjectionParameterSet,
        astro_event_rate: float,
        source_prior: "PriorDict",
        cosmology: Cosmology = DEFAULT_COSMOLOGY,
    ):
        self.cosmology = cosmology
        self.foreground = foreground
        self.rejected = rejected
        self.astro_event_rate = astro_event_rate
        self.injected_volume = self.get_injected_volume(source_prior)

        accepted_weights = self.get_weights(foreground, source_prior)
        rejected_weights = self.get_weights(rejected, source_prior)
        self.weights = accepted_weights
        self.accepted_fraction = accepted_weights.sum() / (
            accepted_weights.sum() + rejected_weights.sum()
        )
        self.fit()

    @property
    def scale_factor(self):
        """Scale factor to convert from rate density to count rate"""
        return (
            self.astro_event_rate
            * self.injected_volume
            * self.accepted_fraction
        )

    def get_weights(
        self, injections: InjectionParameterSet, source_prior: "PriorDict"
    ) -> np.ndarray:
        """
        Weights that reweight injections drawn from the source prior
        to the target redshift distribution, up to a constant.
        """
        z = injections.redshift
        log_target = log_dNdz(z, self.cosmology)
        log_source = np.log(source_prior["redshift"].prob(z))
        return np.exp(log_target - log_source)

    def get_injected_volume(self, source_prior: "PriorDict") -> float:
        """
        Calculate the volume of the universe in which injections were made,
        from the redshift and declination bounds of the source prior.

        Returns:
            The injection volume in cubic gigaparsecs
        """
        redshift, dec = source_prior["redshift"], source_prior["dec"]
        volume = get_astrophysical_volume(
            redshift.minimum,
            redshift.maximum,
            self.cosmology,
            dec_range=(dec.minimum, dec.maximum),
        )
        return volume / 1e9

    def fit(self):
        """
        Fit the foreground model to the data
        """
        raise NotImplementedError

    def __call__(self, stats):
        """
        Defines how to evaluate the foreground model density
        at the given detection statistic
        """
        raise NotImplementedError


class KdeForeground(ForegroundModel):
    """
    Foreground model that uses a KDE to model the detection statistic
    """

    def fit(self):
        self.kde = gaussian_kde(
            self.foreground.detection_statistic, weights=self.weights
        )

    def __call__(self, stats: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the foreground model density at the given detection statistics

        Args:
            stats: Detection statistic(s) at which to evaluate the
                foreground model

        Returns:
            Foreground model density evaluated at the input
            detection statistic(s)
        """
        density = self.kde(stats)
        if isinstance(stats, (float, int)):
            density = density.item()
        return self.scale_factor * density
