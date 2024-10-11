import astropy.cosmology as cosmo
from bilby.core.prior import PriorDict
from bilby.gw.prior import UniformComovingVolume

from priors.priors import uniform_extrinsic, uniform_spin

# default cosmology
COSMOLOGY = cosmo.Planck15


def prior(
    mass_1: float,
    mass_2: float,
    zmax: float,
):
    prior = PriorDict(uniform_extrinsic())
    prior["mass_1"] = mass_1
    prior["mass_2"] = mass_2
    prior["redshift"] = UniformComovingVolume(
        name="redshift",
        minimum=0,
        maximum=zmax,
    )

    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value

    detector_frame_prior = False
    return prior, detector_frame_prior
