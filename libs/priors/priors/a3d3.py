
import astropy.cosmology as cosmo
import numpy as np
from bilby.core.prior import (
    Cosine,
    PriorDict,
    Uniform,
)
from bilby.gw.prior import UniformSourceFrame

from priors.utils import mass_constraints
from priors.priors import uniform_spin, uniform_extrinsic

# default cosmology
COSMOLOGY = cosmo.Planck15

def prior(
    mass_1: float,
    mass_2: float,
):

    prior = PriorDict(uniform_extrinsic())
    prior["mass_1"] = mass_1
    prior["mass_2"] = mass_2
    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2
    )

    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value

    detector_frame_prior = False
    return prior, detector_frame_prior