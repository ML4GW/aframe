# import logging
from typing import List, Optional

import arrakis
import numpy as np
import torch
from online.dataloading.utils import (
    resample,
    build_resample_filter,
)

STRAIN_SAMPLE_RATE = 16384
BLOCK_DURATION = 1 / 16
BLOCK_SIZE = int(BLOCK_DURATION * STRAIN_SAMPLE_RATE)


def data_iterator(
    strain_channels: List[str],
    ifos: List[str],
    sample_rate: float,
    state_channels: Optional[dict[str, str]] = None,
    numtaps: Optional[int] = 60,
) -> torch.Tensor:
    channels = strain_channels + state_channels
    # build resampling filter
    factor = STRAIN_SAMPLE_RATE / sample_rate
    if not factor.is_integer():
        raise ValueError(
            f"Specified sample rate {sample_rate} must "
            f"evenly divide the frame sample rate {STRAIN_SAMPLE_RATE}"
        )
    factor = int(factor)
    b, a = build_resample_filter(factor, numtaps)
    # Need to crop off at least filter size from both sides
    # of the resampled data
    crop_size = numtaps // factor + 1
    crop_length = crop_size / sample_rate

    frame_buffer = np.zeros((len(ifos), 0))
    # slicing will take out 1 second of data from a buffer,
    # removing `crop_size` samples on the right and
    # `BLOCK_SIZE - crop_size` samples on the left.
    slc = slice(-int(crop_size + BLOCK_SIZE), -int(crop_size))
    block_buffer = np.zeros((len(ifos), 0))
    last_ready = [True] * len(ifos)
    for block in arrakis.stream(channels):
        ready = [True] * len(ifos)

        if state_channels is not None:
            for i, channel in enumerate(state_channels):
                state_vector = block[channel].data

                ifo_ready = ((state_vector & 3) == 3).all()
                # Not sure we want to be logging every 16th of a second
                # if not ifo_ready:
                #     logging.warning(f"IFO {channel[:2]} not analysis ready")
                ready[i] &= ifo_ready

        strain_data = np.stack(
            [block[channel].data for channel in strain_channels]
        )
        block_buffer = np.append(block_buffer, strain_data, axis=1)
        dur = block_buffer.shape[-1] / STRAIN_SAMPLE_RATE
        # Need enough time to be able to crop out edge effects
        # from resampling
        if dur >= BLOCK_SIZE + 2 * crop_length:
            x = resample(frame_buffer, factor, b, a)
            x = x[:, slc]
            block_buffer = block_buffer[:, BLOCK_SIZE:]
            yield (
                torch.Tensor(x).double(),
                float(block.t0 - crop_length),
                last_ready,
            )

        last_ready = ready
