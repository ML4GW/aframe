from typing import TYPE_CHECKING, Optional

import numpy as np
from bilby.core.prior import (
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    PowerLaw,
    Sine,
    Uniform,
)
from bilby.gw.prior import BBHPriorDict, UniformSourceFrame

from bbhnet.priors.utils import read_priors_from_file

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology

# Unit names
msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"


class PriorDict(BBHPriorDict):
    def __init__(self, *args, source_frame: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # whether the priors are defined in source frame
        self._source_frame = source_frame

    def sample(self, n: Optional[int] = None, source_frame: bool = False):
        samples = super().sample(n)
        # only rescale if requested frame is different from defined frame
        rescale = self._source_frame != source_frame
        if not rescale:
            return samples

        # convert to requested frame
        factor = 1 + samples["redshift"]
        factor = (1 / factor) if source_frame else factor
        samples["mass_1"] *= factor
        samples["mass_2"] *= factor
        return samples

    # TODO: implement prob method that accounts for jacobian
    # which will be used during importance sampling
    def prob(self):
        raise NotImplementedError


def uniform_extrinsic(source_frame: bool = False) -> PriorDict:
    prior = {}
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = 0
    prior["phase"] = 0

    return prior


def nonspin_bbh(cosmology: Optional["Cosmology"] = None) -> PriorDict:
    prior = PriorDict(dictionary=uniform_extrinsic(), source_frame=False)
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["luminosity_distance"] = UniformSourceFrame(
        0, 2, unit=mpc, name="luminosity_distance", cosmology=cosmology
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
    prior = PriorDict(dictionary=uniform_extrinsic(), source_frame=True)
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
    prior = uniform_extrinsic(source_frame=True)
    event_file = "./event_files/\
        O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_bbh.h5"
    prior |= read_priors_from_file(event_file)

    return prior


def gaussian_masses(
    m1: float, m2: float, sigma: float, cosmology: Optional["Cosmology"] = None
):
    """
    Population prior

    Constructs a gaussian bilby prior for masses.
    Args:
        m1: mean of the Gaussian distribution for mass 1
        m2: mean of the Gaussian distribution for mass 2
        sigma: standard deviation of the Gaussian distribution for both masses

    Returns a BBHpriorDict
    """
    prior_dict = PriorDict(source_frame=True)
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


def log_normal_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: Optional["Cosmology"] = None,
):
    """
    Population prior

    Constructs a log normal bilby prior for masses.
    Args:
        m1: mean of the Log Normal distribution for mass 1
        m2: mean of the Log Normal distribution for mass 2
        sigma: standard deviation for m1 and m2

    Returns a BBHpriorDict
    """
    prior_dict = PriorDict(source_frame=True)
    prior_dict["mass_1"] = LogNormal(name="mass_1", mu=m1, sigma=sigma)
    prior_dict["mass_2"] = LogNormal(name="mass_2", mu=m2, sigma=sigma)
    prior_dict["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    return prior_dict
