from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
from bilby.core.prior import Interped, PriorDict


def mass_ratio_constraint(samples):
    if "mass_1" not in samples.keys() or "mass_2" not in samples.keys():
        raise KeyError("mass_1 and mass_1 must exist to have a mass_ratio")
    out_samples = samples
    out_samples["mass_ratio"] = samples["mass_2"] / samples["mass_1"]
    return out_samples


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


def read_priors_from_file(event_file: Path, *parameters: str) -> PriorDict:
    """
    Reads in a file containing sets of GW parameters and
    returns a set of interpolated priors
    The expected structure is based off the event file from
    here: https://dcc.ligo.org/T2100512

    Args:
        event_file: An hdf5 file containing event parameters
        parameters: Optional, a list of parameters to read from the file

    Returns:
        prior: A PriorDict with priors based on the event file
    """
    prior = PriorDict()
    with h5py.File(event_file, "r") as f:
        events = f["events"]
        field_names = parameters or events.dtype.names
        for name in field_names:
            grid, pdf = pdf_from_events(events[name])
            prior[name] = Interped(grid, pdf, np.min(grid), np.max(grid))

    return prior
