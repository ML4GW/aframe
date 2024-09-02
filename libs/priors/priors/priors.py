import astropy.cosmology as cosmo
import numpy as np
from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    ConditionalUniform,
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    LogUniform,
    PowerLaw,
    PriorDict,
    Sine,
    Triangular,
    Uniform,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame
from cosmology.cosmology import DEFAULT_COSMOLOGY

from priors.utils import (
    mass_condition_powerlaw,
    mass_condition_uniform,
    mass_constraints,
)

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
    prior["inclination"] = Sine()
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


def nonspin_bbh(cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY) -> PriorDict:
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


def spin_bbh(cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY) -> PriorDict:
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
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
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
    prior["mass_1"] = PowerLaw(alpha=-2.35, minimum=5, maximum=100, unit=msun)
    prior["mass_2"] = ConditionalPowerLaw(
        condition_func=mass_condition_powerlaw,
        alpha=1,
        minimum=5,
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


def end_o3_ratesandpops_bns(
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
) -> ConditionalPriorDict:
    """
    Define a Bilby `PriorDict` that matches the BNS distribution used
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
    prior["mass_1"] = Triangular(mode=2.5, minimum=1, maximum=2.5, unit=msun)
    prior["mass_2"] = ConditionalUniform(
        condition_func=mass_condition_uniform,
        minimum=1,
        maximum=2.5,
        unit=msun,
    )
    prior["redshift"] = UniformSourceFrame(
        0, 0.15, name="redshift", cosmology=cosmology
    )
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value
    prior["a_1"] = Uniform(0, 0.4)
    prior["a_2"] = Uniform(0, 0.4)
    detector_frame_prior = False
    return prior, detector_frame_prior


def gaussian_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
):
    """
    Construct a gaussian bilby prior for masses.

    Masses are defined in the source frame.

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

    detector_frame_prior = False
    return prior, detector_frame_prior


def log_normal_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
):
    """
    Construct a log normal bilby prior for masses.

    Masses are defined in the source frame.

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

    prior["mass_1"] = LogNormal(name="mass_1", mu=np.log(m1), sigma=sigma)
    prior["mass_2"] = LogNormal(name="mass_2", mu=np.log(m2), sigma=sigma)
    prior["mass_ratio"] = Constraint(0.02, 1)

    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = False
    return prior, detector_frame_prior


def ringdown_prior(
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
) -> (PriorDict, bool):
    """
    Define a Bilby `PriorDict` containing distributions for ringdown parameters

    Quality, Frequency, and Distance are defined in the detector frame

    Args:
        cosmology: An `astropy` cosmology, used to determine distance sampling

    Returns:
        prior: `
            PriorDict` containing the specified distributions
        detector_frame_prior:
            A boolean indicating if the prior is in the detector frame.
    """
    prior = uniform_extrinsic()
    prior["quality"] = Uniform(8, 20)
    prior["frequency"] = LogUniform(100, 1000)
    prior["distance"] = UniformComovingVolume(
        100, 1000, name="luminosity_distance", cosmology=cosmology
    )

    detector_frame_prior = True
    return prior, detector_frame_prior
