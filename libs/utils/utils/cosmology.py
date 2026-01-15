import astropy.cosmology as cosmo
import numpy as np
from scipy.integrate import quad

DEFAULT_COSMOLOGY = cosmo.Planck15


def volume_element(cosmology, z: float) -> float:
    """
    Calculate the differential comoving volume element per unit redshift.

    Computes dV/dz at a given redshift, accounting for the expansion of
    the universe and normalized by (1+z) for rest-frame calculations.

    Args:
        cosmology (astropy.cosmology.Cosmology): The cosmological model to use
        (e.g., Planck15).
        z (float): Redshift at which to evaluate the volume element.

    Returns:
        float: Differential comoving volume in Mpc^3 per unit redshift
            at redshift z.
    """
    # Get differential comoving volume from cosmology and normalize by (1+z)
    return cosmology.differential_comoving_volume(z).value / (1 + z)


def get_astrophysical_volume(
    zmin: float,
    zmax: float,
    cosmology: cosmo.Cosmology | None = None,
    dec_range: tuple[float, float] | None = None,
) -> float:
    """
    Calculate the astrophysical volume in a redshift and declination range.

    Integrates the differential comoving volume element over a redshift range
    and solid angle (defined by declination range) to get the total accessible
    astrophysical volume for gravitational wave sources.

    Args:
        zmin (float): Minimum redshift of integration range.
        zmax (float): Maximum redshift of integration range.
        cosmology (cosmo.Cosmology, optional): Cosmological model
            to use. If None, uses Planck15. Defaults to None.
        dec_range (tuple[float, float], optional): (declination_min,
            declination_max) in radians. If None, uses full sky
            (dec_range = [-np.pi/2, np.pi/2]). Defaults to None.

    Returns:
        float: Astrophysical volume in Mpc^3 accessible within the specified
            redshift and declination ranges.

    Examples:
        >>> # Full sky volume between z=0 and z=0.1
        >>> vol = get_astrophysical_volume(0, 0.1)

        >>> # Volume in northern hemisphere only
        >>> vol = get_astrophysical_volume(0, 0.1, dec_range=(0, np.pi/2))
    """
    if cosmology is None:
        cosmology = DEFAULT_COSMOLOGY

    # Numerically integrate volume element over redshift range
    volume, _ = quad(lambda z: volume_element(cosmology, z), zmin, zmax)

    # Calculate solid angle from declination range
    if dec_range is not None:
        decmin, decmax = dec_range
        # Convert declinations to zenith angles
        theta_max = np.pi / 2 - decmin
        theta_min = np.pi / 2 - decmax
    else:
        # Full sky coverage
        theta_min, theta_max = 0, np.pi

    # Calculate solid angle in steradians
    omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))

    # Total volume is volume integral * solid angle
    return volume * omega
