import logging
from math import pi
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar

import h5py
import numpy as np
import torch
from train.data_structures import (
    BBHNetWaveformInjection,
    GlitchSampler,
    SignalInverter,
    SignalReverser,
)

from ml4gw.distributions import Cosine, LogNormal, Uniform

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


class MultiInputSequential(torch.nn.Sequential):
    """
    Cheap wrapper around the torch Sequential object
    to support multiple input
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


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
    train_val_start: float,
    train_val_stop: float,
    glitch_prob: float,
    waveform_prob: float,
    glitch_downweight: float,
    swap_frac: float,
    mute_frac: float,
    sample_rate: float,
    highpass: float,
    mean_snr: float,
    std_snr: float,
    min_snr: float,
    trigger_distance: float = 0,
    valid_frac: Optional[float] = None,
):
    # build a glitch sampler from a pre-saved bank of
    # glitches which will randomly insert them into
    # either or both interferometer channels
    glitch_dict = {}
    valid_glitches_list = []

    # calculate the time at which the validation set starts
    valid_start = (1 - valid_frac) * (
        train_val_stop - train_val_start
    ) + train_val_start
    with h5py.File(glitch_dataset, "r") as f:
        for ifo in ifos:
            glitches = f[ifo]["glitches"][:]
            times = f[ifo]["times"][:]

            if valid_frac is not None:
                train_glitches = glitches[times <= valid_start]
                valid_glitches = glitches[times > valid_start]
                logging.info(f"{len(train_glitches)} train glitches for {ifo}")
                logging.info(f"{len(valid_glitches)} valid glitches for {ifo}")
                glitch_dict[ifo] = train_glitches
                valid_glitches_list.append(valid_glitches)
            else:
                logging.info(f"{len(train_glitches)} train glitches for {ifo}")
                glitch_dict[ifo] = glitches
                valid_glitches_list = None

    glitch_inserter = GlitchSampler(
        prob=glitch_prob,
        max_offset=int(trigger_distance * sample_rate),
        **glitch_dict,
    )

    # initiate a waveform sampler from a pre-saved bank
    # of GW waveform polarizations which will randomly
    # project them to inteferometer responses and
    # inject those resposnes into the input data
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]
        if valid_frac is not None:
            signals, valid_signals = split(signals, 1 - valid_frac, 0)
            valid_cross, valid_plus = valid_signals.transpose(1, 0, 2)

            slc = slice(-len(valid_signals), None)
            valid_injector = BBHNetWaveformInjection(
                ifos=ifos,
                dec=f["dec"][slc],
                psi=f["psi"][slc],
                phi=f["ra"][slc],  # no geocent_time recorded, so just use ra
                snr=None,
                sample_rate=sample_rate,
                highpass=highpass,
                trigger_offset=0,
                plus=valid_plus,
                cross=valid_cross,
            )

        else:
            valid_injector = None

    cross, plus = signals.transpose(1, 0, 2)
    # instantiate source parameters as callable
    # distributions which will produce samples
    injector = BBHNetWaveformInjection(
        ifos=ifos,
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        snr=LogNormal(mean_snr, std_snr, min_snr),
        swap_frac=swap_frac,
        mute_frac=mute_frac,
        sample_rate=sample_rate,
        highpass=highpass,
        prob=waveform_prob,
        glitch_prob=glitch_prob,
        downweight=glitch_downweight,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )

    # stack glitch inserter and waveform sampler into
    # a single random augmentation object which will
    # be called at data-loading time (i.e. won't be
    # used on validation data).
    augmenter = MultiInputSequential(
        glitch_inserter, SignalInverter(), SignalReverser(), injector
    )
    return augmenter, valid_glitches_list, valid_injector
