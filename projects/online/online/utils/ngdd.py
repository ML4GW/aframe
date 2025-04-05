# import logging
from typing import List, Optional

import arrakis
import numpy as np
import torch
from scipy.signal import resample

STRAIN_SAMPLE_RATE = 16384
BLOCK_DURATION = 1 / 16
BLOCK_SIZE = int(BLOCK_DURATION * STRAIN_SAMPLE_RATE)


def data_iterator(
    strain_channels: List[str],
    ifos: List[str],
    sample_rate: float,
    state_channels: Optional[dict[str, str]] = None,
) -> torch.Tensor:
    channels = strain_channels + state_channels
    block_buffer = np.zeros((len(ifos), 0))
    # Blocks delivered in 16th of a second intervals
    crop_size = int(sample_rate * BLOCK_DURATION)
    slc = slice(-2 * crop_size, -crop_size)
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
        # Need at least 3 blocks to be able to crop out edge effects
        # from resampling and just yield the middle second
        if dur >= 3 * BLOCK_DURATION:
            x = resample(
                block_buffer,
                int(sample_rate * dur),
                axis=1,
                window="hann",
            )
            x = x[:, slc]
            block_buffer = block_buffer[:, BLOCK_SIZE:]
            yield (
                torch.Tensor(x).double(),
                float(block.t0 - BLOCK_DURATION),
                last_ready,
            )

        last_ready = ready
