from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
from bilby.core.prior import Interped, PriorDict


def chirp_mass(m1, m2):
    """Calculate chirp mass from component masses"""
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)


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


def pdf_from_events(
    param_values: Sequence[float],
    grid_size: int = 100,
    spacing: str = "lin",
) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Estimate the probability distribution of a parameter based on
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


def read_priors_from_file(event_file: Path, *parameters: str) -> PriorDict:
    """
    Reads in a file containing sets of GW parameters and
    returns a set of interpolated priors
    The expected structure is based off the event file from
    here: https://dcc.ligo.org/T2100512

    Args:
        event_file:
            An hdf5 file containing event parameters
        parameters:
            Optional, a list of parameters to read from the file

    Returns:
        prior:
            A PriorDict with priors based on the event file
    """
    prior = PriorDict()
    with h5py.File(event_file, "r") as f:
        events = f["events"]
        field_names = parameters or events.dtype.names
        for name in field_names:
            grid, pdf = pdf_from_events(events[name])
            prior[name] = Interped(grid, pdf, np.min(grid), np.max(grid))

    return prior
