import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from astropy import cosmology as cosmo
from astropy import units as u

if TYPE_CHECKING:
    from astropy.units import Quantity
    from astropy.cosmology import Cosmology
    import bilby

import numpy as np
from scipy.integrate import quad

PI_OVER_TWO = math.pi / 2


def calculate_astrophysical_volume(
    zmin: float,
    zmax: float,
    dec_min: float = -PI_OVER_TWO,
    dec_max: float = PI_OVER_TWO,
    cosmology: "Cosmology" = cosmo.Planck15,
) -> "Quantity":
    """
    Calculates the astrophysical volume over which injections have been made.
    See equation 4) in https://arxiv.org/pdf/1712.00482.pdf

    Args:
        dl_min: minimum distance of injections in Mpc
        dl_max: maximum distance of injections in Mpc
        dec_min: minimum declination of injections in radians
        dec_max: maximum declination of injections in radians
        cosmology: astropy cosmology object

    Returns astropy.Quantity of volume in Mpc^3
    """

    # calculate the angular volume of the sky
    # over which injections have been made
    dec_min, dec_max = dec_min * u.rad, dec_max * u.rad
    theta_max, theta_min = (
        np.pi / 2 - dec_min.value,
        np.pi / 2 - dec_max.value,
    )
    omega = -2 * math.pi * (np.cos(theta_max) - np.cos(theta_min))

    # calculate the volume of the universe
    # over which injections have been made
    integrand = (
        lambda z: 1
        / (1 + z)
        * (cosmology.differential_comoving_volume(z)).value
    )

    volume, _ = quad(integrand, zmin, zmax) * u.Mpc**3 * omega
    return volume


@dataclass
class SensitiveVolumeCalculator:
    """
    Class for calculating sensitive volume metrics using importance sampling.

    Args:
        source:
            Bilby PriorDict of the source distribution
            used to create the injections
        recovered_parameters:
            Dictionary of recovered parameters
        n_injections:
            Number of total injections
        cosmology:
            Astropy Cosmology object used for volume calculation
    """

    source: Callable
    cosmology: "Cosmology" = cosmo.Planck15

    def __post_init__(self):
        self.source, self.detector_frame_prior = self.source(self.cosmology)
        z_prior = self.source["redshift"]
        zmin, zmax = z_prior.minimum, z_prior.maximum

        # if the source distribution has a dec prior,
        # use it to calculate the area on the sky
        # over which injections have been made
        # otherwise, calculate_astrophysical_volume assumes the full sky
        if "dec" in self.source:
            dec_prior = self.source["dec"]
            dec_min, dec_max = dec_prior.minimum, dec_prior.maximum
        else:
            dec_min = dec_max = None

        # calculate the astrophysical volume over
        # which injections have been made.
        self.volume = calculate_astrophysical_volume(
            zmin=zmin,
            zmax=zmax,
            dec_min=dec_min,
            dec_max=dec_max,
            cosmology=self.cosmology,
        )

    def weights(
        self,
        recovered_parameters: Dict[str, np.ndarray],
        target: "bilby.core.prior.PriorDict",
    ):
        """
        Calculate the weights for the samples.
        """
        weights = []
        for sample in recovered_parameters:
            source_frame_sample = sample.copy()
            source_frame_sample["mass_1"] /= 1 + sample["redshift"]
            source_frame_sample["mass_2"] /= 1 + sample["redshift"]
            # calculate the weight for each sample
            # using the source and target distributions
            if self.detector_frame_prior:
                weight = target.prob(sample) / self.source.prob(sample)
            else:
                weight = target.prob(sample) / self.source.prob(
                    source_frame_sample
                )
            weights.append(weight)
        return np.array(weights)

    def __call__(
        self,
        recovered_parameters: List[Dict[str, float]],
        num_injections: int,
        target: Optional["bilby.core.prior.PriorDict"] = None,
    ):
        """
        Calculates the sensitive volume and its uncertainty.
        See equations 8 and 9 in https://arxiv.org/pdf/1904.10879.pdf

        Args:
            target:
                Bilby PriorDict of the target distribution
                used for importance sampling. If None, the source
                distribution is used.

        Returns tuple of (vt, std, n_eff)
        """

        if target is not None:
            weights = self.weights(recovered_parameters, target)
            mu = np.sum(weights) / num_injections
            variance = np.sum(weights**2) / num_injections**2
        else:
            mu = len(recovered_parameters) / num_injections
            variance = len(recovered_parameters) / num_injections**2

        variance -= mu**2 / num_injections
        v = mu * self.volume
        variance *= v**2

        std = np.sqrt(variance)
        n_eff = v**2 / variance
        return v.value, std.value, n_eff.value