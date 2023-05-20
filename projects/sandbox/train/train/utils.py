import logging
from math import pi
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar

import h5py
import numpy as np
import torch

from train.augmentor import AframeBatchAugmentor
from train.data_structures import GlitchSampler, SnrRescaler, SnrSampler

import ml4gw.gw as gw
from ml4gw.distributions import Cosine, Uniform

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


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


def prepare_augmentation(
    glitch_dataset: Path,
    waveform_dataset: Path,
    ifos: List[str],
    train_start: float,
    train_stop: float,
    glitch_prob: float,
    waveform_prob: float,
    glitch_downweight: float,
    swap_frac: float,
    mute_frac: float,
    sample_rate: float,
    highpass: float,
    max_min_snr: float,
    min_min_snr: float,
    max_snr: float,
    snr_alpha: float,
    snr_decay_steps: float,
    invert_prob: Optional[float] = 0.5,
    reverse_prob: Optional[float] = 0.5,
    trigger_distance: float = 0,
    valid_frac: Optional[float] = None,
):
    # build a glitch sampler from a pre-saved bank of
    # glitches which will randomly insert them into
    # either or both interferometer channels
    glitch_dict = {}
    valid_glitches = []

    # calculate the time at which the validation set starts
    full_duration = train_stop - train_start
    train_duration = (1 - valid_frac) * full_duration
    valid_start = train_start + train_duration
    with h5py.File(glitch_dataset, "r") as f:
        for ifo in ifos:
            glitches = f[ifo]["glitches"][:]
            times = f[ifo]["times"][:]

            if valid_frac is not None:
                train_glitches = glitches[times <= valid_start]
                ifo_glitches = glitches[times > valid_start]
                logging.info(f"{len(train_glitches)} train glitches for {ifo}")
                logging.info(f"{len(ifo_glitches)} valid glitches for {ifo}")
                glitch_dict[ifo] = train_glitches
                valid_glitches.append(ifo_glitches)
            else:
                logging.info(f"{len(train_glitches)} train glitches for {ifo}")
                glitch_dict[ifo] = glitches
                valid_glitches = None

    glitch_sampler = GlitchSampler(
        prob=glitch_prob,
        max_offset=int(trigger_distance * sample_rate),
        **glitch_dict,
    )

    tensors, vertices = gw.get_ifo_geometry(*ifos)
    # perform train/val split of waveforms,
    # and compute fixed validation responses
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]
        if valid_frac is not None:
            signals, valid_signals = split(signals, 1 - valid_frac, 0)
            valid_cross, valid_plus = valid_signals.transpose(1, 0, 2)
            valid_cross, valid_plus = torch.Tensor(valid_cross), torch.Tensor(
                valid_plus
            )
            slc = slice(-len(valid_signals), None)
            dec, psi, phi = f["dec"][slc], f["psi"][slc], f["ra"][slc]
            dec, psi, phi = (
                torch.Tensor(dec),
                torch.Tensor(psi),
                torch.Tensor(phi),
            )
            valid_responses = gw.compute_observed_strain(
                dec,
                psi,
                phi,
                detector_tensors=tensors,
                detector_vertices=vertices,
                sample_rate=sample_rate,
                plus=valid_plus,
                cross=valid_cross,
            )
        else:
            valid_responses = None

    cross, plus = signals.transpose(1, 0, 2)
    waveform_duration = cross.shape[-1] / sample_rate

    # construct the augmentor that will be used at training time
    # to sample waveforms, rescale snrs, insert glitches, perform
    # background strain inversions and flips, etc.
    snr = SnrSampler(
        max_min_snr, min_min_snr, max_snr, snr_alpha, snr_decay_steps
    )
    rescaler = SnrRescaler(
        len(ifos),
        sample_rate,
        waveform_duration,
        highpass,
    )

    snr = SnrSampler(max_mean_snr, min_mean_snr, std_snr, snr_decay_steps)

    augmentor = AframeBatchAugmentor(
        ifos,
        sample_rate,
        mute_frac,
        swap_frac,
        glitch_downweight,
        waveform_prob,
        glitch_sampler,
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        trigger_distance=trigger_distance,
        snr=snr,
        rescaler=rescaler,
        invert_prob=invert_prob,
        reverse_prob=reverse_prob,
        cross=cross,
        plus=plus,
    )

    return augmentor, valid_glitches, valid_responses
