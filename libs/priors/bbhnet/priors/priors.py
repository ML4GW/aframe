from typing import TYPE_CHECKING, Optional

import numpy as np
from bilby.core.prior import (
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
)
from bilby.gw.prior import UniformSourceFrame

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology

from bbhnet.priors.utils import mass_constraints, read_priors_from_file

# Unit names
msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"


def uniform_extrinsic() -> PriorDict:
    prior = PriorDict(conversion_function=mass_constraints)
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = 0
    prior["phase"] = 0

    return prior


def nonspin_bbh(cosmology: Optional["Cosmology"] = None) -> PriorDict:
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["redshift"] = UniformSourceFrame(
        0, 0.5, name="redshift", cosmology=cosmology
    )
    prior["psi"] = 0
    prior["a_1"] = 0
    prior["a_2"] = 0
    prior["tilt_1"] = 0
    prior["tilt_2"] = 0
    prior["phi_12"] = 0
    prior["phi_jl"] = 0

    detector_frame_prior = True
    return prior, detector_frame_prior


def spin_bbh(cosmology: Optional["Cosmology"] = None) -> PriorDict:
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["redshift"] = UniformSourceFrame(
        0, 0.5, name="redshift", cosmology=cosmology
    )
    prior["psi"] = 0
    prior["a_1"] = Uniform(0, 0.998)
    prior["a_2"] = Uniform(0, 0.998)
    prior["tilt_1"] = Sine(unit=rad)
    prior["tilt_2"] = Sine(unit=rad)
    prior["phi_12"] = 0
    prior["phi_jl"] = 0

    detector_frame_prior = True
    return prior, detector_frame_prior


def end_o3_ratesandpops(
    cosmology: Optional["Cosmology"] = None,
) -> PriorDict:
    prior = uniform_extrinsic()
    prior["mass_1"] = PowerLaw(alpha=-2.35, minimum=2, maximum=100, unit=msun)
    prior["mass_2"] = PowerLaw(alpha=1, minimum=2, maximum=100, unit=msun)
    prior["mass_ratio"] = Constraint(0.02, 1)
    prior["chirp_mass"] = Constraint(10, 100, unit=msun)
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

    detector_frame_prior = False
    return prior, detector_frame_prior


def power_law_dip_break():
    prior = uniform_extrinsic()
    event_file = "./event_files/\
        O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_bbh.h5"
    prior |= read_priors_from_file(event_file)

    detector_frame_prior = False
    return prior, detector_frame_prior


def gaussian_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: Optional["Cosmology"] = None,
):
    """
    Constructs a gaussian bilby prior for masses.
    Args:
        m1: mean of the Gaussian distribution for mass 1
        m2: mean of the Gaussian distribution for mass 2
        sigma: standard deviation of the Gaussian distribution for both masses

    Returns a PriorDict
    """
    prior = PriorDict()
    prior["mass_1"] = Gaussian(name="mass_1", mu=m1, sigma=sigma)
    prior["mass_2"] = Gaussian(name="mass_2", mu=m2, sigma=sigma)
    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = True
    return prior, detector_frame_prior


def log_normal_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: Optional["Cosmology"] = None,
):
    """
    Constructs a log normal bilby prior for masses.
    Args:
        m1: mean of the Log Normal distribution for mass 1
        m2: mean of the Log Normal distribution for mass 2
        sigma: standard deviation for m1 and m2

    Returns a PriorDict
    """
    prior = PriorDict()
    prior["mass_1"] = LogNormal(name="mass_1", mu=m1, sigma=sigma)
    prior["mass_2"] = LogNormal(name="mass_2", mu=m2, sigma=sigma)
    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = True
    return prior, detector_frame_prior
