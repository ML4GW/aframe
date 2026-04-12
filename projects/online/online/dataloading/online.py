import logging
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from gwpy.timeseries import TimeSeries

from online.dataloading.utils import (
    GWF_SAMPLE_RATE,
    PATH_LIKE,
    build_resample_filter,
    fname_re,
    is_gwf,
    resample,
)

BLOCK_DURATION = 1
BLOCK_SIZE = int(BLOCK_DURATION * GWF_SAMPLE_RATE)


def get_prefix(datadir: Path):
    if not datadir.exists():
        raise FileNotFoundError(f"No data directory '{datadir}'")

    fnames = map(str, datadir.iterdir())
    matches = map(fname_re.search, fnames)
    matches = list(filter(is_gwf, matches))

    if len(matches) == 0:
        raise ValueError(f"No valid .gwf files in data directory '{datadir}'")

    t0 = max([int(i.group("start")) for i in matches])
    prefixes = {i.group("prefix") for i in matches}
    if len(prefixes) > 1:
        raise ValueError(
            f"Too many prefixes {list(prefixes)} in data directory '{datadir}'"
        )

    durations = {i.group("duration") for i in matches}
    if len(durations) > 1:
        raise ValueError(
            f"Too many lengths {list(durations)} in data directory '{datadir}'"
        )
    return list(prefixes)[0], int(list(durations)[0]), t0


def reset_t0(datadir, last_t0):
    tick = time.time()
    while True:
        matches = [fname_re.search(i.name) for i in datadir.iterdir()]
        t0s = np.array([int(i.group("start")) for i in matches if is_gwf(i)])
        if t0s.size > 0:
            t0 = max(t0s)
            logging.info(f"Resetting timestamp to {t0}")
            return t0

        time.sleep(1)
        elapsed = (time.time() - tick) // 1
        if not elapsed % 10:
            logging.info(
                f"No new frames available since timestamp {last_t0}, "
                f"elapsed time {elapsed}s"
            )


def data_iterator(
    datadir: Path,
    channels: list[str],
    ifos: list[str],
    sample_rate: float,
    ifo_suffix: str = None,
    state_channels: dict[str, str] | None = None,
    timeout: float | None = None,
    numtaps: int | None = 60,
) -> Generator[tuple[torch.Tensor, float, list[bool]], None, None]:
    if ifo_suffix is not None:
        ifo_dir = "_".join([ifos[0], ifo_suffix])
    else:
        ifo_dir = ifos[0]
    prefix, length, t0 = get_prefix(datadir / ifo_dir)
    middle = "_".join(prefix.split("_")[1:])

    # build resampling filter
    factor = GWF_SAMPLE_RATE / sample_rate
    if not factor.is_integer():
        raise ValueError(
            f"Specified sample rate {sample_rate} must "
            f"evenly divide the frame sample rate {GWF_SAMPLE_RATE}"
        )
    factor = int(factor)
    b, a = build_resample_filter(factor, numtaps)
    # Need to crop off at least half the filter size from
    # both sides of the resampled data. Stick with powers of
    # 2 to avoid issues coverting between time and samples.
    crop_size = 2 ** np.ceil(np.log2((numtaps / 2) / factor))
    crop_length = crop_size / sample_rate

    frame_buffer = np.zeros((len(ifos), 0))
    # slicing will take out 1 second of data from a buffer,
    # removing `crop_size` samples on the right and
    # `BLOCK_DURATION * sample_rate - crop_size` samples on the left.
    resampled_block_size = BLOCK_DURATION * sample_rate
    slc = slice(-int(crop_size + resampled_block_size), -int(crop_size))
    last_ready = [True] * len(ifos)
    while True:
        frames = []
        logging.debug(f"Reading frames from timestamp {t0}")

        ready = [True] * len(ifos)
        for i, (ifo, channel) in enumerate(zip(ifos, channels, strict=True)):
            prefix = f"{ifo[0]}-{ifo}_{middle}"
            if ifo_suffix is not None:
                ifo_dir = "_".join([ifo, ifo_suffix])
            else:
                ifo_dir = ifo
            fname = datadir / ifo_dir / f"{prefix}-{t0}-{length}.gwf"

            tick = time.time()
            while not fname.exists():
                tock = time.time()
                if timeout is not None and (tock - tick > timeout):
                    logging.warning(
                        f"Couldn't find frame file {fname} after {timeout}s"
                    )

                    yield None, t0, [False] * len(ifos)

                    frame_buffer = np.zeros((len(ifos), 0))
                    last_ready = [False] * len(ifos)
                    t0 = reset_t0(datadir / ifo, t0 - length)
                    break
            else:
                # we never broke, therefore the filename exists,
                # so read the strain data as well as its state
                # vector to see if it's analysis ready
                x = read_channel(fname, f"{channel}")
                frames.append(x.value)

                # if state channels were specified,
                # check that the 3rd bit is on.
                # If left as `None` set ifo_ready
                # to True by default.
                # TODO: parameterize bitmask
                ifo_ready = True
                if state_channels is not None:
                    state_channel = state_channels[ifo]
                    state_vector = read_channel(fname, state_channel)
                    ifo_ready = ((state_vector.value & 3) == 3).all()

                # some useful logging
                # for when ifos enter and exit
                # analyis ready mode
                if not ifo_ready:
                    if last_ready[i]:
                        logging.info(f"IFO {ifo} exiting analysis ready mode")
                    else:
                        logging.debug(f"IFO {ifo} not analysis ready")
                else:
                    if not last_ready[i]:
                        logging.info(f"IFO {ifo} entering analysis ready mode")

                # mark this ifos readiness in array
                ready[i] &= ifo_ready

                # continue so that we don't break the ifo for-loop
                continue

            # if we're here, the filename didnt' exist and
            # we broke when resetting t0, so don't bother
            # to return any data
            break
        else:
            logging.debug("Read successful")

            frame = np.stack(frames)
            frame_buffer = np.append(frame_buffer, frame, axis=1)
            dur = frame_buffer.shape[-1] / GWF_SAMPLE_RATE
            # Need enough time to be able to crop out edge effects
            # from resampling
            if dur >= BLOCK_DURATION + 2 * crop_length:
                x = resample(frame_buffer, factor, b, a)
                x = x[:, slc]
                frame_buffer = frame_buffer[:, BLOCK_SIZE:]
                # yield last_ready, which corresponds to
                # the data quality bits of the previous second
                # of data, i.e. the middle second of the
                # buffer that is being yielded as well
                yield (
                    torch.Tensor(x.copy()).double(),
                    t0 - crop_length,
                    last_ready,
                )

            last_ready = ready
            t0 += length


def read_channel(fname: PATH_LIKE, channel: str, num_retries: int = 3):
    """
    Read a channel from a frame file, retrying if the read fails
    and handling common errors that can occur when reading frame files
    """
    for i in range(num_retries):
        try:
            x = TimeSeries.read(fname, channel=channel)
        except ValueError as e:
            if str(e) == (
                "Cannot generate TimeSeries with 2-dimensional data"
            ):
                logging.warning(
                    f"Channel {channel} from file {fname} got corrupted "
                    f"and was read as 2D, attempting reread {i + 1}"
                )
                time.sleep(1e-1)
                continue
            elif str(e).startswith("Creation of unknown checksum type"):
                time.sleep(1e-1)
                continue
            else:
                raise
        except RuntimeError as e:
            if str(e).startswith("Failed to read the core"):
                logging.warning(
                    f"Channel {channel} from file {fname} had "
                    f"corrupted header, attempting reread {i + 1}"
                )
                time.sleep(2e-1)
                continue
            elif str(e).startswith("Missing FrEndOfFile structure"):
                logging.warning(
                    f"File {fname} was missing FrEndOfFile structure, "
                    f"attempting reread {i + 1}"
                )
                time.sleep(1e-1)
                continue
            else:
                raise

        if len(x) != x.sample_rate.value:
            logging.warning(
                f"Channel {channel} in file {fname} got corrupted with "
                f"length {len(x)}, attempting reread {i + 1}"
            )
            del x
            time.sleep(1e-1)
            continue

        return x
    else:
        raise ValueError(f"Failed to read channel {channel} in file {fname}")
