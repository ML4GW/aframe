import numpy as np
from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    Constraint,
    Cosine,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame


def nonspin_bbh():

    prior_dict = PriorDict()
    prior_dict["mass_1"] = Uniform(
        name="mass_1", minimum=5, maximum=100, unit=r"$M_{\odot}$"
    )
    prior_dict["mass_2"] = Uniform(
        name="mass_2", minimum=5, maximum=100, unit=r"$M_{\odot}$"
    )
    prior_dict["mass_ratio"] = Constraint(
        name="mass_ratio", minimum=0.2, maximum=5.0
    )
    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=100, maximum=3000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = 0
    prior_dict["psi"] = 0
    prior_dict["phase"] = 0
    prior_dict["a_1"] = 0
    prior_dict["a_2"] = 0
    prior_dict["tilt_1"] = 0
    prior_dict["tilt_2"] = 0
    prior_dict["phi_12"] = 0
    prior_dict["phi_jl"] = 0

    return prior_dict


def end_o3_ratesandpops():

    prior_dict = ConditionalPriorDict()
    prior_dict["mass_1"] = PowerLaw(
        name="mass_1", alpha=-2.35, minimum=2, maximum=100, unit=r"$M_{\odot}"
    )

    def condition_func(reference_params, mass_1):
        return dict(
            alpha=reference_params["alpha"],
            minimum=reference_params["minimum"],
            maximum=mass_1,
        )

    prior_dict["mass_2"] = ConditionalPowerLaw(
        name="mass_2",
        condition_func=condition_func,
        alpha=1,
        minimum=2,
        maximum=100,
        unit=r"$M_{\odot}",
    )
    prior_dict["luminosity_distance"] = UniformComovingVolume(
        name="luminosity_distance", minimum=100, maximum=15000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = 0
    prior_dict["psi"] = 0
    prior_dict["phase"] = 0
    prior_dict["a_1"] = Uniform(name="a_1", minimum=0, maximum=0.998)
    prior_dict["a_2"] = Uniform(name="a_2", minimum=0, maximum=0.998)
    prior_dict["tilt_1"] = Sine(name="tilt_1", unit="rad")
    prior_dict["tilt_2"] = Sine(name="tilt_2", unit="rad")
    prior_dict["phi_12"] = Uniform(
        name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["phi_jl"] = 0

    return prior_dict
