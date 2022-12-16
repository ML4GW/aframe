from collections import OrderedDict
from math import pi
from pathlib import Path
from typing import Optional, Tuple, TypeVar

import h5py
import numpy as np
import torch

from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.waveform_injection import BBHNetWaveformInjection
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
    if isinstance(X, np.ndarray):
        return np.split(X, [size], axis=axis)
    else:
        splits = [size, X.shape[axis] - size]
        return torch.split(X, splits, dim=axis)


def prepare_augmentation(
    glitch_dataset: Path,
    waveform_dataset: Path,
    glitch_prob: float,
    waveform_prob: float,
    sample_rate: float,
    highpass: float,
    mean_snr: float,
    std_snr: float,
    min_snr: Optional[float] = None,
    trigger_distance: float = 0,
    valid_frac: Optional[float] = None,
):
    augmentation_layers = OrderedDict()

    # build a glitch sampler from a pre-saved bank of
    # glitches which will randomly insert them into
    # either or both interferometer channels
    with h5py.File(glitch_dataset, "r") as f:
        h1_glitches = f["H1_glitches"][:]
        l1_glitches = f["L1_glitches"][:]

    if valid_frac is not None:
        h1_glitches, valid_h1_glitches = split(h1_glitches, 1 - valid_frac, 0)
        l1_glitches, valid_l1_glitches = split(l1_glitches, 1 - valid_frac, 0)
        valid_glitches = [valid_h1_glitches, valid_l1_glitches]

    augmentation_layers["glitch_inserter"] = GlitchSampler(
        prob=glitch_prob,
        max_offset=int(trigger_distance * sample_rate),
        H1=h1_glitches,
        L1=l1_glitches,
    )

    # initiate a waveform sampler from a pre-saved bank
    # of GW waveform polarizations which will randomly
    # project them to inteferometer responses and
    # inject those resposnes into the input data
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]
        if valid_frac is not None:
            signals, valid_signals = split(signals, 1 - valid_frac, 0)
            valid_plus, valid_cross = valid_signals.transpose(1, 0, 2)

            slc = slice(-len(valid_signals), None)
            valid_injector = BBHNetWaveformInjection(
                ifos=["H1", "L1"],
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

    plus, cross = signals.transpose(1, 0, 2)

    # instantiate source parameters as callable
    # distributions which will produce samples
    augmentation_layers["injector"] = BBHNetWaveformInjection(
        ifos=["H1", "L1"],
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        snr=LogNormal(mean_snr, std_snr, min_snr),
        sample_rate=sample_rate,
        highpass=highpass,
        prob=waveform_prob,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )

    # stack glitch inserter and waveform sampler into
    # a single random augmentation object which will
    # be called at data-loading time (i.e. won't be
    # used on validation data).
    augmenter = MultiInputSequential(augmentation_layers)

    if valid_frac is not None:
        return augmenter, valid_glitches, valid_injector
    return augmenter, None, None
