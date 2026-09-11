import logging
from math import lcm

import numpy as np
import torch
from arrakis import Client, Time
from online.dataloading.utils import (
    resample,
    build_resample_filter,
)


def stream_channels(
    strain_channels: list[str],
    ifos: list[str],
    state_channels: dict[str, str] | None = None,
) -> list[str]:
    """The full list of channels that will be streamed"""
    # Prevents modifying in-place
    channels = list(strain_channels)
    if state_channels is not None:
        channels += [state_channels[ifo] for ifo in ifos]
    return channels


def get_block_duration(
    channels: list[str], metadata: dict | None = None
) -> float:
    """
    The cadence at which the server will deliver blocks of
    data, which is determined by the least common multiple
    of the individual stride of each channel.
    """
    if not metadata:
        metadata = Client().describe(channels)
    strides = [metadata[channel].stride for channel in channels]
    # Strides are returned in nanoseconds
    return lcm(*strides) / Time.SECONDS


def get_strain_sample_rate(strain_channels: list[str], metadata: dict) -> int:
    """
    Finds the sample rate of the strain channels and checks that they agree
    """
    sample_rates = {
        metadata[channel].sample_rate for channel in strain_channels
    }
    if len(sample_rates) > 1:
        raise ValueError(
            f"Strain channels {strain_channels} have different sample rates: "
            f"{sample_rates}"
        )
    return int(sample_rates.pop())


def data_iterator(
    strain_channels: list[str],
    ifos: list[str],
    sample_rate: float,
    state_channels: dict[str, str] | None = None,
    numtaps: int | None = 60,
) -> torch.Tensor:
    channels = stream_channels(strain_channels, ifos, state_channels)

    client = Client()
    metadata = client.describe(channels)
    strain_sample_rate = get_strain_sample_rate(strain_channels, metadata)
    block_duration = get_block_duration(channels, metadata)

    # build resampling filter
    factor = strain_sample_rate / sample_rate
    if not factor.is_integer():
        raise ValueError(
            f"Specified sample rate {sample_rate} must "
            f"evenly divide the frame sample rate {strain_sample_rate}"
        )
    factor = int(factor)
    b, a = build_resample_filter(factor, numtaps)
    # Need to crop off at least half the filter size from
    # both sides of the resampled data. Stick with powers of
    # 2 to avoid issues coverting between time and samples.
    crop_size = 2 ** np.ceil(np.log2((numtaps / 2) / factor))
    crop_length = crop_size / sample_rate

    # slicing will take out 1 second of data from a buffer,
    # removing `crop_size` samples on the right and
    # `block_duration * sample_rate - crop_size` samples on the left.
    resampled_block_size = block_duration * sample_rate
    slc = slice(-int(crop_size + resampled_block_size), -int(crop_size))
    block_buffer = np.zeros((len(ifos), 0))
    last_ready = [True] * len(ifos)

    # Arrakis will skip ahead if the stream has fallen far enough behind
    # real-time, so we need to track the expected t0 to know when there's
    # a discontinuous jump
    expected_t0 = None

    blocks = client.stream(channels)
    for block in blocks:
        # Check if the expected t0 differs by more than half a sample
        discontinuous = expected_t0 is not None and (
            abs(block.t0 - expected_t0) > 0.5 / strain_sample_rate
        )
        if block.has_gaps or discontinuous:
            if discontinuous:
                logging.warning(
                    f"Stream jumped from {expected_t0} to {block.to}, "
                    "resetting states"
                )
            else:
                logging.warning(
                    f"Gap in stream at {block.t0}, resetting states"
                )

            yield None, float(block.t0), [False] * len(ifos)

            block_buffer = np.zeros((len(ifos), 0))
            last_ready = [False] * len(ifos)
            expected_t0 = None
            continue

        ready = [True] * len(ifos)
        if state_channels is not None:
            for i, ifo in enumerate(ifos):
                state_vector = block[state_channels[ifo]].data

                ifo_ready = ((state_vector & 3) == 3).all()

                if ifo_ready and not last_ready[i]:
                    logging.info(f"{ifo} entering analysis-ready mode")
                elif not ifo_ready and last_ready[i]:
                    logging.info(f"{ifo} exiting analysis-ready mode")

                ready[i] &= ifo_ready

        strain_data = np.stack(
            [block[channel].data for channel in strain_channels]
        )
        block_buffer = np.append(block_buffer, strain_data, axis=1)
        block_size = strain_data.shape[-1]
        expected_t0 = block.t0 + block_size / strain_sample_rate

        dur = block_buffer.shape[-1] / strain_sample_rate
        # Need enough time to be able to crop out edge effects
        # from resampling
        if dur >= block_duration + 2 * crop_length:
            x = resample(block_buffer, factor, b, a)
            x = x[:, slc]
            block_buffer = block_buffer[:, block_size:]
            yield (
                torch.Tensor(x.copy()).double(),
                float(block.t0 - crop_length),
                last_ready,
            )

        last_ready = ready
