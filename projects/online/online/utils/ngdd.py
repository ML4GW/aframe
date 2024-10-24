import logging
from typing import List

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
) -> torch.Tensor:
    state_channels = [f"{ifo}:GDS-CALIB_STATE_VECTOR" for ifo in ifos]
    channels = strain_channels + state_channels
    block_buffer = np.zeros((len(ifos), 0))
    # Blocks delivered in 16th of a second intervals
    slc = slice(-2 * BLOCK_SIZE, -BLOCK_SIZE)
    last_ready = True
    for block in arrakis.stream(channels):
        ready = True
        for channel in state_channels:
            state_vector = block[channel].data
            ifo_ready = ((state_vector.value & 3) == 3).all()
            # Not sure we want to be logging every 16th of a second
            if not ifo_ready:
                logging.warning(f"IFO {channel[:2]} not analysis ready")
            ready &= ifo_ready

        strain_data = np.stack([block[channel] for channel in strain_channels])
        block_buffer = np.append(block_buffer, strain_data, axis=1)
        # Need at least 3 blocks to be able to crop out edge effects
        # from resampling and just yield the middle second
        if block_buffer.shape[-1] >= 3 * BLOCK_SIZE:
            x = resample(
                block_buffer,
                int(sample_rate * BLOCK_DURATION),
                axis=1,
                window="hann",
            )
            x = x[:, slc]
            block_buffer = block_buffer[:, STRAIN_SAMPLE_RATE:]
            yield torch.Tensor(
                x
            ).double(), block.t0 - BLOCK_DURATION, last_ready

        last_ready = ready
