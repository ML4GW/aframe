import logging
from math import ceil
from typing import Iterable

import numpy as np
import torch
from train.utils import split


def make_validation_dataset(
    background: np.ndarray,
    glitches: Iterable[np.ndarray],
    waveforms: torch.Tensor,
    kernel_length: float,
    stride: float,
    sample_rate: float,
    batch_size: int,
    glitch_frac: float,
    device: str,
):
    # redefine args in terms of number of samples, infer
    # some downstream parameters that depend on them
    kernel_size = int(kernel_length * sample_rate)
    stride_size = int(stride * sample_rate)
    num_kernels = (background.shape[-1] - kernel_size) // stride_size + 1
    num_kernels = int(num_kernels)
    num_ifos = len(background)
    if len(glitches) != num_ifos:
        raise ValueError(
            "Number of glitch tensors {} doesn't match number "
            "of interferometers {}".format(len(glitches), num_ifos)
        )

    # 1. Create pre-computed kernels of pure background
    # slice our background so that it has an integer number of
    # windows, then add dummy dimensions since unfolding only
    # works on 4D tensors
    background = background[:, : num_kernels * stride_size + kernel_size]
    background = torch.Tensor(background).view(1, num_ifos, 1, -1)

    # fold out into windows up front
    background = torch.nn.functional.unfold(
        background, (1, num_kernels), dilation=(1, stride_size)
    )

    # some reshape magic having to do with how the
    # unfold op orders things. Don't worry about it
    background = background.reshape(num_ifos, num_kernels, kernel_size)
    background = background.transpose(1, 0)

    # 2. Copy these windows and insert glitches into the
    # interferometer channels to make a glitch dataset
    # begin by setting aside the first `glitch_frac` of
    # each channel to be placed in coincident kernels
    h1_glitches, l1_glitches = map(torch.Tensor, glitches)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1
    num_coinc = int(glitch_frac * num_glitches / (1 + glitch_frac))

    h1_coinc, h1_glitches = split(h1_glitches, num_coinc / num_h1, 0)
    l1_coinc, l1_glitches = split(l1_glitches, num_coinc / num_l1, 0)
    coinc = torch.stack([h1_coinc, l1_coinc], axis=1)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1 + num_coinc

    # if we need to create duplicates of some of our
    # background to make this work, figure out how many
    repeats = ceil(num_glitches / len(background))
    glitch_background = background.repeat(repeats, 1, 1)
    glitch_background = glitch_background[:num_glitches]

    # now insert the glitches
    start = h1_glitches.shape[-1] // 2 - kernel_size // 2
    stop = start + kernel_size
    glitch_background[:num_h1, 0] = h1_glitches[:, start:stop]
    glitch_background[num_h1:-num_coinc, 1] = l1_glitches[:, start:stop]
    glitch_background[-num_coinc:] = coinc[:, :, start:stop]

    # 3. create a tensor of background with waveforms injected
    repeats = ceil(len(waveforms) / len(background))
    waveform_background = background.repeat(repeats, 1, 1)
    waveform_background = waveform_background[: len(waveforms)]

    start = waveforms.shape[-1] // 2 - kernel_size // 2
    stop = start + kernel_size
    waveform_background += waveforms[:, :, start:stop]

    # concatenate everything into a single tensor
    # and create the associated labels
    X = torch.concat([background, glitch_background, waveform_background])
    y = torch.zeros((len(X),))
    y[-len(waveform_background) :] = 1

    logging.info("Performing validation on:")
    logging.info(f"    {len(background)} windows of background")
    logging.info(f"    {num_h1} H1 glitches")
    logging.info(f"    {num_l1} L1 glitches")
    logging.info(f"    {num_coinc} coincident glitches")
    logging.info(f"    {len(waveforms)} injected waveforms")
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        pin_memory_device=device,
    )
