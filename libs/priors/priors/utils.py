from typing import Dict, List


def chirp_mass(m1, m2):
    """Calculate chirp mass from component masses"""
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)


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


def mass_condition_powerlaw(reference_params, mass_1):
    """
    Return a dictionary that can be interpreted by Bilby's
    `ConditionalPowerLaw` to set the maximum value of `mass_2`
    to be whatever was drawn for `mass_1`
    """
    return dict(
        alpha=reference_params["alpha"],
        minimum=reference_params["minimum"],
        maximum=mass_1,
    )


def mass_constraints(samples):
    """
    Return a dictionary with new keys `mass_ratio` and `chirp_mass`
    so that Bilby `Constraint`s on these keys will be effected
    """
    if "mass_1" not in samples or "mass_2" not in samples:
        raise KeyError("mass_1 and mass_1 must exist to have a mass_ratio")
    samples["mass_ratio"] = samples["mass_2"] / samples["mass_1"]
    samples["chirp_mass"] = chirp_mass(samples["mass_1"], samples["mass_2"])
    return samples


def transpose(d: Dict[str, List]):
    """Turn a dict of lists into a list of dicts"""
    return [dict(zip(d, col)) for col in zip(*d.values())]
