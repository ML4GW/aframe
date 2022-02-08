from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import convolve

from bbhnet import io


def boxcar_filter(t, y, window_length: float = 1.0):
    sample_rate = 1 / (t[1] - t[0])

    window_size = int(window_length * sample_rate)
    window = np.ones((window_size,)) / window_size

    mf = convolve(y, window, mode="valid")
    t = t[window_size - 1 :]
    return t, mf


def analyze_segment(
    fnames: List[str],
    window_length: float = 1,
    norm_seconds: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze a segment of time-contiguous BBHNet outputs

    Compute matched filter outputs on a stretch
    of frame files that are assumed to be contiguous
    and ordered in time. Matched filters are computed
    as the average over the the last `window_length`
    seconds of data, optionally normalized by the mean
    and standard deviation of the previous `norm_seconds`
    seconds worth of data.

    Args:
        fnames:
            Filenames to of HDF5 files containing neural
            network outputs and the initial timestamps of
            the corresponding input kernels in the keys
            `"out"` and `"GPSstart"` respectively.
            Should be formatted so that they end with the pattern
            '<initial GPS timestamp>-<length in seconds>.hdf5'.
        window_length:
            The length of time, in seconds, over which previous
            network outputs should be averaged to produce
            "matched filter" outputs.
        norm_seconds:
            The number of seconds before each matched filter
            window over which to compute the mean and
            standard deviation of network outputs, which will
            be used to normalize filter outputs. If left as
            `None`, filter outputs won't be normalized.
    Returns:
        Array of timestamps corresponding to the
            _end_ of the input kernel that would produce
            the corresponding network output and matched
            filter output
        Array of raw neural network outputs for each timestamp
        Array of matched filter outputs for each timestamp
    """

    # if we specified a normalization period, ensure
    # that we have at least 50% of that period to analyze
    if norm_seconds is not None:
        min_seconds = 1.5 * norm_seconds
        total_seconds = 0
        for f in fnames:
            try:
                total_seconds += float(io.fname_re.search(f).group("length"))
            except AttributeError:
                raise ValueError(f"Filename {f} not properly formatted")

        if total_seconds < min_seconds:
            raise ValueError(
                "Segment from filenames {} has length {}s, "
                "analysis requires at least {}s of data".format(
                    fnames, total_seconds, min_seconds
                )
            )

    # read in and process all our data
    # add one second to our timestamps
    # to make them equate to when samples
    # enter our "field of vision"
    t, y = zip(*map(io.read_fname, io.filter_and_sort_files(fnames)))
    t = np.concatenate(t) + 1
    y = np.concatenate(y)
    t, mf = boxcar_filter(t, y, window_length=window_length)

    if norm_seconds is not None:
        # compute the mean and standard deviation of
        # the last `norm_seconds` seconds of data
        # compute the standard deviation by the
        # sigma^2 = E[x^2] - E^2[x] trick
        _, shifts = boxcar_filter(t, y, norm_seconds)
        _, sqs = boxcar_filter(t, y ** 2, norm_seconds)
        scales = np.sqrt(sqs - shifts ** 2)

        # slice all our arrays from the latest
        # possible time forward, to make sure that
        # we're dealing with strictly past data
        mf = mf[-len(shifts) :]
        t = t[-len(shifts) :]
        y = y[-len(shifts) :]
        mf = (mf - shifts) / scales
    else:
        y = y[-len(mf) :]
    return t, y, mf
