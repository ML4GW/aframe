import random
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar

import h5py
import numpy as np
import torch

import ml4gw.gw as gw

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split(X: Tensor, frac: float, axis: int) -> Tuple[Tensor, Tensor]:
    """
    Split an array into two parts along the given axis
    by an amount specified by `frac`. Generic to both
    numpy arrays and torch Tensors.
    """

    size = int(frac * X.shape[axis])
    # Catches fp error that sometimes happens when size should be an exact int
    # Is there a better way to do this?
    if np.abs(frac * X.shape[axis] - size - 1) < 1e-10:
        size += 1

    if isinstance(X, np.ndarray):
        return np.split(X, [size], axis=axis)
    else:
        splits = [size, X.shape[axis] - size]
        return torch.split(X, splits, dim=axis)


def _sort_key(fname: Path):
    """
    Return the intified value of the string between
    the final two hypens of the given `Path`
    """
    return int(fname.stem.split("-")[-2])


def get_background_fnames(data_dir: Path, min_valid_duration: float):
    """
    Return list of background filenames in `data_dir` sorted
    by GPS start time of the data, which is assumed to be
    given in the file name between the final two hypens.
    See `_sort_key`.
    """
    fnames = data_dir.glob("*.hdf5")
    fnames = sorted(fnames, key=_sort_key)
    durations = [int(fname.stem.split("-")[-1]) for fname in fnames]

    valid_fnames = []
    valid_duration = 0
    while valid_duration < min_valid_duration:
        fname, duration = fnames.pop(-1), durations.pop(-1)
        valid_duration += duration
        valid_fnames.append(fname)
    return list(fnames), valid_fnames


def get_background(fnames: Iterable[Path]):
    """
    Load the background from the given HDF5 files
    """
    data = []
    for fname in fnames:
        background = []
        with h5py.File(fname, "r") as f:
            ifos = list(f.keys())
            for ifo in ifos:
                hoft = f[ifo][:]
                background.append(hoft)
        data.append(np.stack(background))
    return np.stack(data)


def get_waveforms(
    waveform_dataset: Path,
    ifos: List[str],
    sample_rate: float,
    valid_frac: float,
):
    """
    Load the training waveforms, and if `valid_frac` is not
    `None`, perform the train/val split and compute fixed
    validation responses
    """
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]

        if valid_frac is not None:
            signals, valid_signals = split(signals, 1 - valid_frac, 0)

            valid_cross, valid_plus = valid_signals.transpose(1, 0, 2)
            slc = slice(-len(valid_signals), None)
            dec, psi, phi = f["dec"][slc], f["psi"][slc], f["ra"][slc]

            # project the validation waveforms to IFO
            # responses up front since we don't care
            # about sampling sky parameters
            tensors, vertices = gw.get_ifo_geometry(*ifos)
            valid_responses = gw.compute_observed_strain(
                torch.Tensor(dec),
                torch.Tensor(psi),
                torch.Tensor(phi),
                detector_tensors=tensors,
                detector_vertices=vertices,
                sample_rate=sample_rate,
                plus=torch.Tensor(valid_plus),
                cross=torch.Tensor(valid_cross),
            )
            return signals, valid_responses
    return signals, None
