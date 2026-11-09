import logging
from math import lcm

import numpy as np
import torch
from arrakis import Client, Time
from online.dataloading.utils import (
    span_ready,
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


def check_replay_id(replay_id: str) -> None:
    """Check that `replay_id` is one registered on the Arrakis server"""
    replays = Client().replays()
    if replay_id not in replays:
        raise ValueError(
            f"Unknown replay ID {replay_id}. "
            f"Available replays: {sorted(replays)}"
        )


def get_block_duration(
    channels: list[str],
    metadata: dict | None = None,
    replay_id: str | None = None,
) -> float:
    """
    The cadence at which the server will deliver blocks of
    data, which is determined by the least common multiple
    of the individual stride of each channel.
    """
    if not metadata:
        metadata = Client().describe(channels, replay_id=replay_id)
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


def gapped_ifos(
    block,
    ifos: list[str],
    strain_channels: list[str],
    state_channels: dict[str, str] | None = None,
) -> list[bool]:
    """
    Which interferometers have a gap in either their strain channel or
    state vector
    """
    return [
        block[strain_channel].has_gaps
        or (
            block[state_channels[ifo]].has_gaps
            if state_channels is not None
            else False
        )
        for ifo, strain_channel in zip(ifos, strain_channels, strict=True)
    ]


def data_iterator(
    strain_channels: list[str],
    ifos: list[str],
    sample_rate: float,
    state_channels: dict[str, str] | None = None,
    numtaps: int | None = 60,
    replay_id: str | None = None,
) -> torch.Tensor:
    channels = stream_channels(strain_channels, ifos, state_channels)

    client = Client()
    metadata = client.describe(channels, replay_id=replay_id)
    strain_sample_rate = get_strain_sample_rate(strain_channels, metadata)
    block_duration = get_block_duration(channels, metadata, replay_id)

    # build resampling filter
    factor = strain_sample_rate / sample_rate
    if not factor.is_integer():
        raise ValueError(
            f"Specified sample rate {sample_rate} must "
            f"evenly divide the strain sample rate {strain_sample_rate}"
        )
    factor = int(factor)
    b, a = build_resample_filter(factor, numtaps)
    # filtfilt runs the filter forwards and then backwards, so its
    # effective kernel is `numtaps` samples wide on either side of
    # each output. Crop that much from both sides of the resampled
    # data. Stick with powers of 2 to avoid issues converting
    # between time and samples.
    crop_size = 2 ** np.ceil(np.log2(numtaps / factor))
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

    # whether the buffer is being refilled after a gap
    refilling = False

    blocks = client.stream(channels, replay_id=replay_id)
    for block in blocks:
        # Check if the expected t0 differs by more than half a sample
        discontinuous = expected_t0 is not None and (
            abs(block.t0 - expected_t0) > 0.5 / strain_sample_rate
        )
        gapped = gapped_ifos(block, ifos, strain_channels, state_channels)
        # The main search loop requires the ifos to be HL or HLV, so the first
        # two are always H1 and L1. The search can't run without both, so a gap
        # or a jump in either resets the states. We can run without V1, so a
        # gap or a jump there just marks V1 as not ready.
        if any(gapped[:2]) or discontinuous:
            if discontinuous:
                logging.warning(
                    f"Stream jumped from {expected_t0} to {block.t0}, "
                    "resetting states"
                )
            else:
                logging.warning(
                    f"Gap in stream at {block.t0}, resetting states"
                )

            yield None, float(block.t0 - crop_length), [False] * len(ifos)

            block_buffer = np.zeros((len(ifos), 0))
            last_ready = [False] * len(ifos)
            expected_t0 = None
            refilling = True
            continue

        # Initialize the readiness array based on which ifos don't contain
        # gaps. Only V1 can still contain gaps at this point due to the check
        # above, so this can only be (assuming HLV) `[True, True, True]` or
        # `[True, True, False]`.
        ready = [not gap for gap in gapped]
        if state_channels is not None:
            for i, ifo in enumerate(ifos):
                state_vector = block[state_channels[ifo]].data

                # An ifo is ready when the two lowest bits of its state
                # vector are both 1. 3 is 0b11, so bitwise-anding with 3
                # results in 3 only when the two lowest bits are on.
                # A gapped state vector can't show readiness, and checking
                # one that's fully masked gives `masked` rather than a bool
                ifo_ready = not gapped[i] and bool(
                    ((state_vector & 3) == 3).all()
                )

                if ifo_ready and not last_ready[i]:
                    logging.info(f"{ifo} entering analysis-ready mode")
                elif not ifo_ready and last_ready[i]:
                    logging.info(f"{ifo} exiting analysis-ready mode")

                ready[i] &= ifo_ready

        # Replace any gapped V1 data with zeros. Only V1 can be gapped here,
        # since neither H1 nor L1 can be after the check above.
        strain_data = np.stack(
            [
                np.ma.filled(block[channel].data, 0)
                for channel in strain_channels
            ]
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
                span_ready(last_ready, ready),
            )
            refilling = False
        elif refilling:
            # the first block after a gap can't be analyzed because the
            # buffer hasn't refilled yet. Yielding as we do above with data
            # gaps makes sure that time is recorded as part of the gap.
            yield None, float(block.t0 - crop_length), [False] * len(ifos)

        last_ready = ready
