import astropy.cosmology as cosmo
import astropy.units as u
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
    rate_index: float = 0.0,
    rate_redshift: float = 0.0,
) -> float:
    """
    Calculate the astrophysical volume in a redshift and declination range.

    Integrates the differential comoving volume element over a redshift range
    and solid angle (defined by declination range) to get the total accessible
    astrophysical volume for gravitational wave sources.

    If the merger rate density evolves with redshift, the volume element
    is weighted by the rate density relative to its value at `rate_redshift`,
    `((1 + z) / (1 + rate_redshift)) ** rate_index`. The rate density at
    `rate_redshift` times this volume is then the rate of mergers.

    Args:
        zmin (float): Minimum redshift of integration range.
        zmax (float): Maximum redshift of integration range.
        cosmology (cosmo.Cosmology, optional): Cosmological model
            to use. If None, uses Planck15. Defaults to None.
        dec_range (tuple[float, float], optional): (declination_min,
            declination_max) in radians. If None, uses full sky
            (dec_range = [-np.pi/2, np.pi/2]). Defaults to None.
        rate_index (float, optional): Power-law index of the merger rate
            density in (1 + z). Defaults to 0, a rate density that is
            constant in redshift.
        rate_redshift (float, optional): Redshift at which the merger
            rate density is quoted. Defaults to 0.

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

    def rate_weighted_element(z):
        evolution = ((1 + z) / (1 + rate_redshift)) ** rate_index
        return volume_element(cosmology, z) * evolution

    # Numerically integrate volume element over redshift range
    volume, _ = quad(rate_weighted_element, zmin, zmax)

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


def log_dNdz(
    z: np.ndarray,
    cosmology: cosmo.Cosmology,
    num_points: int = 10_000,
    rate_index: float = 0.0,
    rate_redshift: float = 0.0,
) -> np.ndarray:
    """
    Log redshift density of sources whose merger rate per comoving
    volume and source-frame time goes as
    `((1 + z) / (1 + rate_redshift)) ** rate_index`. With the default
    index of 0 the sources are uniform in comoving volume and
    source-frame time.

    For efficiency, the differential comoving volume is evaluated on
    a grid of `num_points` redshifts spanning the minimum and maximum
    of `z`, linearly spaced in log(z). This is used to compute a grid
    of log(dN/dz), which is then interpolated for the requested redshifts.

    Args:
        z: source redshifts
        cosmology: astropy cosmology that sets the comoving volume element
        num_points: Number of points to use to create the redshift grid.
        rate_index: Power-law index of the merger rate density in (1 + z)
        rate_redshift: Redshift at which the merger rate density is quoted

    Returns:
        Log of 4 pi dVc/dz / (1 + z) times the rate density relative to
        its value at `rate_redshift`, with dVc/dz in units of Gpc^3 / sr
    """
    if len(z) == 0:
        return np.empty(0)
    log_z = np.log(z)
    log_z_grid = np.linspace(log_z.min(), log_z.max(), num_points)
    z_grid = np.exp(log_z_grid)

    log_dNdV = rate_index * (np.log(1 + z_grid) - np.log(1 + rate_redshift))
    log_dVdz = np.log(4 * np.pi) + np.log(
        cosmology.differential_comoving_volume(z_grid)
        .to(u.Gpc**3 / u.sr)
        .value
    )
    log_time_dilation = -np.log(1 + z_grid)
    log_dNdz_grid = log_dNdV + log_dVdz + log_time_dilation

    return np.interp(log_z, log_z_grid, log_dNdz_grid)
