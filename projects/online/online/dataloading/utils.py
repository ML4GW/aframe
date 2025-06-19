import re
from typing import Union
from scipy import signal
from gwpy.signal import filter_design
import numpy as np
from pathlib import Path

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


def parse_frame_name(fname: PATH_LIKE) -> tuple[str, int, int]:
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


def is_gwf(match):
    return match is not None and match.group("suffix") == "gwf"
