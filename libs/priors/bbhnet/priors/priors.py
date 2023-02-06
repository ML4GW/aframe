from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import h5py
import numpy as np
from bilby.core.prior import (
    Constraint,
    Cosine,
    Gaussian,
    Interped,
    LogNormal,
    PowerLaw,
    Sine,
    Uniform,
)
from bilby.gw.prior import BBHPriorDict, UniformSourceFrame

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology

# Unit names
msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"


def pdf_from_events(
    param_values: Sequence[float],
    grid_size: int = 100,
    spacing: str = "lin",
) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Estimates the probability distribution of a parameter based on
    a list of sampled values. Currently does this by just creating
    a histogram of the values, but might consider doing a KDE in
    the future

    Args:
        param_values:
            A list of parameter values drawn from the distribution
            to be estimated
        grid_size:
            The number of points at which to estimate the pdf
        spacing:
            The spacing type of the grid, either linear or log

    Returns:
        grid:
            The values at which the pdf was estimated
        pdf:
            The estimated pdf
    """
    param_min = np.min(param_values)
    param_max = np.max(param_values)
    if spacing == "lin":
        bins = np.linspace(param_min, param_max, grid_size + 1)
        grid = (bins[:-1] + bins[1:]) / 2
    elif spacing == "log":
        min_exp = np.log10(param_min)
        max_exp = np.log10(param_max)
        bins = np.logspace(min_exp, max_exp, grid_size + 1)
        grid = np.sqrt(bins[:-1] * bins[1:])
    else:
        raise ValueError("Spacing must be either 'lin' or 'log'")

    pdf, _ = np.histogram(param_values, bins, density=True)

    return grid, pdf


def read_priors_from_file(event_file: Path, *parameters: str) -> BBHPriorDict:
    """
    Reads in a file containing sets of GW parameters and
    returns a set of interpolated priors
    The expected structure is based off the event file from
    here: https://dcc.ligo.org/T2100512

    Args:
        event_file: An hdf5 file containing event parameters
        parameters: Optional, a list of parameters to read from the file

    Returns:
        prior: A BBHPriorDict with priors based on the event file
    """
    prior = BBHPriorDict()
    with h5py.File(event_file, "r") as f:
        events = f["events"]
        field_names = parameters or events.dtype.names
        for name in field_names:
            grid, pdf = pdf_from_events(events[name])
            prior[name] = Interped(grid, pdf, np.min(grid), np.max(grid))

    return prior


def uniform_extrinsic() -> BBHPriorDict:
    prior = BBHPriorDict()
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = 0
    prior["phase"] = 0

    return prior


def nonspin_bbh(cosmology: Optional["Cosmology"] = None) -> BBHPriorDict:
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["luminosity_distance"] = UniformSourceFrame(
        0, 2, unit=mpc, name="redshift", cosmology=cosmology
    )
    prior["psi"] = 0
    prior["a_1"] = 0
    prior["a_2"] = 0
    prior["tilt_1"] = 0
    prior["tilt_2"] = 0
    prior["phi_12"] = 0
    prior["phi_jl"] = 0

    return prior


def end_o3_ratesandpops(cosmology: "Cosmology") -> BBHPriorDict:
    """
    `population prior`
    """
    prior = uniform_extrinsic()
    prior["mass_1"] = PowerLaw(alpha=-2.35, minimum=2, maximum=100, unit=msun)
    prior["mass_2"] = PowerLaw(alpha=1, minimum=2, maximum=100, unit=msun)
    prior["mass_ratio"] = Constraint(0.02, 1)
    prior["redshift"] = UniformSourceFrame(
        0, 2, name="redshift", cosmology=cosmology
    )
    prior["psi"] = 0
    prior["a_1"] = Uniform(0, 0.998)
    prior["a_2"] = Uniform(0, 0.998)
    prior["tilt_1"] = Sine(unit=rad)
    prior["tilt_2"] = Sine(unit=rad)
    prior["phi_12"] = Uniform(0, 2 * np.pi)
    prior["phi_jl"] = 0

    return prior


def power_law_dip_break():
    prior = uniform_extrinsic()
    event_file = "./event_files/\
        O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_bbh.h5"
    prior |= read_priors_from_file(event_file)

    return prior


def gaussian_masses(
    m1: float, m2: float, sigma: float, cosmology: Optional["Cosmology"] = None
):
    """
    Constructs a gaussian bilby prior for masses.
    Args:
        m1: mean of the Gaussian distribution for mass 1
        m2: mean of the Gaussian distribution for mass 2
        sigma: standard deviation of the Gaussian distribution for both masses

    Returns a BBHpriorDict
    """
    prior_dict = BBHPriorDict()
    prior_dict["mass_1"] = Gaussian(name="mass_1", mu=m1, sigma=sigma)
    prior_dict["mass_2"] = Gaussian(name="mass_2", mu=m2, sigma=sigma)
    prior_dict["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    return prior_dict


def log_normal_masses(m1: float, m2: float, sigma: float = 2):
    """
    `Population prior`

    Constructs a log normal bilby prior for masses.
    Args:
        m1: mean of the Log Normal distribution for mass 1
        m2: mean of the Log Normal distribution for mass 2
        sigma: standard deviation for m1 and m2

    Returns a BBHpriorDict
    """
    prior_dict = BBHPriorDict()
    prior_dict["mass_1"] = LogNormal(name="mass_1", mu=m1, sigma=sigma)
    prior_dict["mass_2"] = LogNormal(name="mass_2", mu=m2, sigma=sigma)
    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=100, maximum=3000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    return prior_dict
