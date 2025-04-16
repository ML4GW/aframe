import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from scipy import signal
from gwpy.signal import filter_design

PATH_LIKE = Union[str, Path]
GWF_SAMPLE_RATE = 16384

patterns = {
    "prefix": "[a-zA-Z0-9_:-]+",
    "start": "[0-9]{10}",
    "duration": "[1-9][0-9]*",
    "suffix": "(gwf)|(hdf5)|(h5)",
}
groups = {k: f"(?P<{k}>{v})" for k, v in patterns.items()}
pattern = "{prefix}-{start}-{duration}.{suffix}".format(**groups)
fname_re = re.compile(pattern)


# reproduce exact parameters of
# gwpys TimeSeries.resample() method,
# when downsampling by integer factor
def build_resample_filter(factor: int):
    n = 60
    filt = signal.firwin(n + 1, 1.0 / factor, window="hamming")
    _, filt = filter_design.parse_filter(filt)
    b, a = filt
    return b, a


def resample(data: np.ndarray, factor: int, b: float, a: float):
    return signal.filtfilt(b, a, data, axis=1)[:, :: int(factor)]


def parse_frame_name(fname: PATH_LIKE) -> Tuple[str, int, int]:
    """Use the name of a frame file to infer its initial timestamp and length

    Expects frame names to follow a standard nomenclature
    where the name of the frame file ends {prefix}_{timestamp}-{length}.gwf

    Args:
        fname: The name of the frame file
    Returns:
        The prefix of the frame file name
        The initial GPS timestamp of the frame file
        The length of the frame file in seconds
    """

    if isinstance(fname, Path):
        fname = fname.name

    match = fname_re.search(fname)
    if match is None:
        raise ValueError(f"Could not parse frame filename {fname}")

    prefix, start, duration, *_ = match.groups()
    return prefix, int(start), int(duration)


def _is_gwf(match):
    return match is not None and match.group("suffix") == "gwf"


def get_prefix(datadir: Path):
    if not datadir.exists():
        raise FileNotFoundError(f"No data directory '{datadir}'")

    fnames = map(str, datadir.iterdir())
    matches = map(fname_re.search, fnames)
    matches = list(filter(_is_gwf, matches))

    if len(matches) == 0:
        raise ValueError(f"No valid .gwf files in data directory '{datadir}'")

    t0 = max([int(i.group("start")) for i in matches])
    prefixes = {i.group("prefix") for i in matches}
    if len(prefixes) > 1:
        raise ValueError(
            "Too many prefixes {} in data directory '{}'".format(
                list(prefixes), datadir
            )
        )

    durations = {i.group("duration") for i in matches}
    if len(durations) > 1:
        raise ValueError(
            "Too many lengths {} in data directory '{}'".format(
                list(durations), datadir
            )
        )
    return list(prefixes)[0], int(list(durations)[0]), t0


def reset_t0(datadir, last_t0):
    tick = time.time()
    while True:
        matches = [fname_re.search(i.name) for i in datadir.iterdir()]
        t0s = np.array([int(i.group("start")) for i in matches if _is_gwf(i)])
        if t0s.size > 0:
            t0 = max(t0s)
            logging.info(f"Resetting timestamp to {t0}")
            return t0

        time.sleep(1)
        elapsed = (time.time() - tick) // 1
        if not elapsed % 10:
            logging.info(
                "No new frames available since timestamp {}, "
                "elapsed time {}s".format(last_t0, elapsed)
            )


def data_iterator(
    datadir: Path,
    channels: List[str],
    ifos: List[str],
    sample_rate: float,
    ifo_suffix: str = None,
    state_channels: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> torch.Tensor:
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
    b, a = build_resample_filter(factor)

    frame_buffer = np.zeros((len(ifos), 0))
    # slice corresponds to middle second of
    # a 3 second buffer; the middle second is
    # yielded at each step to mitigate resampling
    # edge effects
    slc = slice(-int(2 * sample_rate), -int(sample_rate))
    last_ready = [True] * len(ifos)
    while True:
        frames = []
        logging.debug(f"Reading frames from timestamp {t0}")

        ready = [True] * len(ifos)
        for i, (ifo, channel) in enumerate(zip(ifos, channels)):
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
                        "Couldn't find frame file {} after {}s".format(
                            fname, timeout
                        )
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
            # Need at least 3 seconds to be able to crop out edge effects
            # from resampling and just yield the middle second
            if dur >= 3:
                x = resample(frame_buffer, factor, b, a)
                x = x[:, slc]
                frame_buffer = frame_buffer[:, GWF_SAMPLE_RATE:]
                # yield last_ready, which corresponds to
                # the data quality bits of the previous second
                # of data, i.e. the middle second of the
                # buffer that is being yielded as well
                yield torch.Tensor(x.copy()).double(), t0 - 1, last_ready

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
                    "Channel {} from file {} got corrupted and was "
                    "read as 2D, attempting reread {}".format(
                        channel, fname, i + 1
                    )
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
                    "Channel {} from file {} had corrupted header, "
                    "attempting reread {}".format(channel, fname, i + 1)
                )
                time.sleep(2e-1)
                continue
            elif str(e).startswith("Missing FrEndOfFile structure"):
                logging.warning(
                    "File {} was missing FrEndOfFile structure, "
                    "attempting reread {}".format(fname, i + 1)
                )
                time.sleep(1e-1)
                continue
            else:
                raise

        if len(x) != x.sample_rate.value:
            logging.warning(
                "Channel {} in file {} got corrupted with "
                "length {}, attempting reread {}".format(
                    channel, fname, len(x), i + 1
                )
            )
            del x
            time.sleep(1e-1)
            continue

        return x
    else:
        raise ValueError(
            "Failed to read channel {} in file {}".format(channel, fname)
        )
