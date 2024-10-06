from typing import Optional

import astropy.cosmology as cosmo
import numpy as np
from scipy.integrate import quad

DEFAULT_COSMOLOGY = cosmo.Planck15


def volume_element(cosmology, z):
    return cosmology.differential_comoving_volume(z).value / (1 + z)


def get_astrophysical_volume(
    zmin: float,
    zmax: float,
    cosmology,
    dec_range: Optional[tuple[float, float]] = None,
):
    """
    Return the astrophysical volume in Mpc^3
    """
    volume, _ = quad(lambda z: volume_element(cosmology, z), zmin, zmax)
    if dec_range is not None:
        decmin, decmax = dec_range
        theta_max = np.pi / 2 - decmin
        theta_min = np.pi / 2 - decmax
    else:
        theta_min, theta_max = 0, np.pi
    omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
    return volume * omega
