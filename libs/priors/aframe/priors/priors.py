from typing import Dict

import astropy.cosmology as cosmo
import numpy as np
from astropy import units
from astropy.cosmology import z_at_value
from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    ConditionalUniform,
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

from aframe.priors.utils import (
    mass_condition_powerlaw,
    mass_constraints,
    read_priors_from_file,
)

# default cosmology
COSMOLOGY = cosmo.Planck15

# Unit names
msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"


def uniform_extrinsic() -> PriorDict:
    """
    Define a Bilby `PriorDict` containing distributions that are
    uniform over the allowed ranges of extrinsic binary black hole
    parameters.
    """
    prior = PriorDict()
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = Sine()
    prior["phase"] = Uniform(0, 2 * np.pi)

    return prior


def uniform_spin() -> PriorDict:
    """
    Define a Bilby `PriorDict` containing distributions that are
    uniform over the allowed ranges of binary black hole spin
    parameters.
    """
    prior = PriorDict()
    prior["psi"] = Uniform(0, np.pi)
    prior["a_1"] = Uniform(0, 0.998)
    prior["a_2"] = Uniform(0, 0.998)
    prior["tilt_1"] = Sine(unit=rad)
    prior["tilt_2"] = Sine(unit=rad)
    prior["phi_12"] = Uniform(0, 2 * np.pi)
    prior["phi_jl"] = Uniform(0, 2 * np.pi)
    return prior


def nonspin_bbh(cosmology: cosmo.Cosmology = COSMOLOGY) -> PriorDict:
    """
    Define a Bilby `PriorDict` that describes a reasonable population
    of non-spinning binary black holes

    Masses are defined in the detector frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
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


def spin_bbh(cosmology: cosmo.Cosmology = COSMOLOGY) -> PriorDict:
    """
    Define a Bilby `PriorDict` that describes a reasonable population
    of spin-aligned binary black holes

    Masses are defined in the detector frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
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
    cosmology: cosmo.Cosmology = COSMOLOGY,
) -> ConditionalPriorDict:
    """
    Define a Bilby `PriorDict` that matches the distributions used
    by the LIGO Rates and Populations group for pipeline searches
    at the end of the third observing run.

    Masses are defined in the source frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = ConditionalPriorDict(uniform_extrinsic())
    prior["mass_1"] = PowerLaw(alpha=-2.35, minimum=10, maximum=100, unit=msun)
    prior["mass_2"] = ConditionalPowerLaw(
        condition_func=mass_condition_powerlaw,
        alpha=1,
        minimum=10,
        maximum=100,
        unit=msun,
    )
    prior["redshift"] = UniformComovingVolume(
        0, 2, name="redshift", cosmology=cosmology
    )
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value
    detector_frame_prior = False
    return prior, detector_frame_prior


def mass_condition_uniform(reference_params, mass_1):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalUniform` to set the maximum value of `mass_2`
    to be whatever was drawn for `mass_1`
    """
    return dict(
        minimum=reference_params["minimum"],
        maximum=mass_1,
    )


def mdc_prior(cosmology: cosmo.Cosmology = COSMOLOGY, method="constrain"):
    """
    Define a Bilby `PriorDict` that matches the distributions used
    by in a machine learning mock data challenge.
    See https://github.com/gwastro/ml-mock-data-challenge-1

    Masses are defined in the detector frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling
        method:
            Determines the sampling method for mass 2. If "constrain",
            the sampling rejects any samples with `mass_2 > mass_1`.
            If "condition", mass 2 is sampled with an upper limit
            set by the mass 1 sample

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    if method == "constrain":
        prior = PriorDict(conversion_function=mass_constraints)
        prior["mass_2"] = Uniform(7, 50, unit=msun)
        prior["mass_ratio"] = Constraint(0.02, 1)
    elif method == "condition":
        prior = ConditionalPriorDict()
        prior["mass_2"] = ConditionalUniform(
            condition_func=mass_condition_uniform,
            minimum=7,
            maximum=50,
            unit=msun,
        )
    else:
        raise ValueError(f"Unknown MDC sampling method {method}")

    prior["mass_1"] = Uniform(7, 50, unit=msun)
    prior["redshift"] = UniformComovingVolume(
        0, 2, name="redshift", cosmology=cosmology
    )
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value

    extrinsic_prior = uniform_extrinsic()
    for key, value in extrinsic_prior.items():
        prior[key] = value

    detector_frame_prior = True
    return prior, detector_frame_prior


def power_law_dip_break():
    """
    Create a Bilby `PriorDict` from a set of sampled parameters
    following the Power Law + Dip + Break model,
    see https://dcc.ligo.org/LIGO-T2100512/public

    Masses are defined in the source frame.

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
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
    cosmology: cosmo.Cosmology = COSMOLOGY,
):
    """
    Construct a gaussian bilby prior for masses.

    Masses are defined in the detector frame.

    Args:
        m1:
            Mean of the Gaussian distribution for mass 1
        m2:
            Mean of the Gaussian distribution for mass 2
        sigma:
            Standard deviation of the Gaussian distribution for both masses
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = PriorDict(conversion_function=mass_constraints)
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


def get_log_normal_params(mean, std):
    """
    Calculate the mean and standard deviation of the normal
    distribution associated with the lognormal distribution
    defined by the given mean and standard deviation
    """
    sigma = np.log((std / mean) ** 2 + 1) ** 0.5
    mu = 2 * np.log(mean / (mean**2 + std**2) ** 0.25)
    return mu, sigma


def log_normal_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: cosmo.Cosmology = COSMOLOGY,
):
    """
    Construct a log normal bilby prior for masses.

    Masses are defined in the detector frame.

    Args:
        m1:
            Mean of the Log Normal distribution for mass 1
        m2:
            Mean of the Log Normal distribution for mass 2
        sigma:
            Standard deviation for m1 and m2
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = PriorDict(conversion_function=mass_constraints)

    mu1, sigma1 = get_log_normal_params(m1, sigma)
    mu2, sigma2 = get_log_normal_params(m2, sigma)
    prior["mass_1"] = LogNormal(name="mass_1", mu=mu1, sigma=sigma1)
    prior["mass_2"] = LogNormal(name="mass_2", mu=mu2, sigma=sigma2)
    prior["mass_ratio"] = Constraint(0.02, 1)

    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = True
    return prior, detector_frame_prior


# The below two functions are for direct comparison with the methodology
# used in the ML MDC paper: https://arxiv.org/abs/2209.11146
def mdc_prior_chirp_distance():
    """
    Same as `mdc_prior` above, but sampling in chirp distance rather than
    redshift. This function always uses the "constraint" method of sampling
    mass 2.
    """
    prior = PriorDict(conversion_function=mass_constraints)
    prior["mass_2"] = Uniform(7, 50, unit=msun)
    prior["mass_ratio"] = Constraint(0.00, 1)
    prior["mass_1"] = Uniform(7, 50, unit=msun)
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value

    extrinsic_prior = uniform_extrinsic()
    for key, value in extrinsic_prior.items():
        prior[key] = value

    prior["chirp_distance"] = PowerLaw(2, 130, 350)
    detector_frame_prior = True

    return prior, detector_frame_prior


def convert_mdc_prior_samples(
    samples: Dict[str, np.ndarray], cosmology: cosmo.Cosmology = COSMOLOGY
):
    """
    Convert samples produced by the `mdc_prior_chirp_distance` prior into
    chirp mass, luminosity distance and redshift.
    """
    samples["chirp_mass"] = (samples["mass_1"] * samples["mass_2"]) ** 0.6 / (
        samples["mass_1"] + samples["mass_2"]
    ) ** 0.2

    fiducial = 1.4 / (2 ** (1 / 5))
    factor = (fiducial / samples["chirp_mass"]) ** (5 / 6)
    samples["luminosity_distance"] = samples["chirp_distance"] / factor
    samples["redshift"] = z_at_value(
        cosmology.luminosity_distance,
        samples["luminosity_distance"] * units.Mpc,
    ).value
    return samples
