import numpy as np
from astropy.cosmology import Cosmology, Planck15
from ledger.events import RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from scipy.integrate import quad
from scipy.stats import gaussian_kde


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
        cosmology:
            The cosmology to use when calculating the injected volume
    """

    def __init__(
        self,
        foreground: RecoveredInjectionSet,
        rejected: InjectionParameterSet,
        astro_event_rate: float,
        cosmology: Cosmology = Planck15,
    ):
        self.cosmology = cosmology
        self.foreground = foreground
        self.rejected = rejected
        self.astro_event_rate = astro_event_rate
        self.injected_volume = self.get_injected_volume()
        self.fit()

    @property
    def total_injections(self):
        return len(self.foreground) + len(self.rejected)

    @property
    def scale_factor(self):
        return (
            self.astro_event_rate
            * self.injected_volume
            * len(self.foreground)
            / self.total_injections
        )

    def _volume_element(self, z):
        return self.cosmology.differential_comoving_volume(z).value / (1 + z)

    def get_injected_volume(self) -> float:
        """
        Calculate the volume of the universe in which injections were made.

        Returns:
            The injection volume in cubic gigaparsecs
        """
        zmin = min(
            self.rejected.redshift.min(), self.foreground.redshift.min()
        )
        zmax = max(
            self.rejected.redshift.max(), self.foreground.redshift.max()
        )
        dec_min = min(self.rejected.dec.min(), self.foreground.dec.min())
        dec_max = max(self.rejected.dec.max(), self.foreground.dec.max())

        volume, _ = quad(lambda z: self._volume_element(z), zmin, zmax)
        theta_max = np.pi / 2 - dec_min
        theta_min = np.pi / 2 - dec_max
        omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
        return volume * omega / 1e9

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
        self.kde = gaussian_kde(self.foreground.detection_statistic)

    def __call__(self, stats):
        density = self.kde(stats)
        if isinstance(stats, (float, int)):
            density = density.item()
        return self.scale_factor * density
