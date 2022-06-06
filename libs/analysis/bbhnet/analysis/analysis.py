from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from bbhnet.analysis.integrators import boxcar_filter

if TYPE_CHECKING:
    from bbhnet.analysis.integrators import Integrator
    from bbhnet.analysis.normalizers import Normalizer
    from bbhnet.io.timeslides import Segment


def integrate(
    segment: "Segment",
    kernel_length: float = 1,
    window_length: Optional[float] = None,
    integrator: "Integrator" = boxcar_filter,
    normalizer: Optional["Normalizer"] = None,
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
        segment:
            Segment of contiguous HDF5 files to analyze
        kernel_length:
            The length of time, in seconds, of the input kernel
            to BBHNet used to produce the outputs being analyzed
        window_length:
            The length of time, in seconds, over which previous
            network outputs should be averaged to produce
            "matched filter" outputs. If left as `None`, it will
            default to the same length as the kernel length.
        integrator:
            Callable which maps from an array of raw neural network
            and a integer window size to an array of integrated
            outputs. Default `boxcar_filter` just performs simple
            uniform integration
        normalizer:
            Callable with a `.fit` method to fit a background of
            raw neural network outputs for normalizing integrated
            outputs
    Returns:
        Array of timestamps corresponding to the
            _end_ of the input kernel that would produce
            the corresponding network output and matched
            filter output
        Array of raw neural network outputs for each timestamp
        Array of matched filter outputs for each timestamp
    """

    # read in all the data for a given segment
    # TODO: should we make the dataset name an argument?
    y, t = segment.load("out")
    sample_rate = 1 / (t[1] - t[0])
    window_length = window_length or kernel_length
    window_size = int(window_length * sample_rate)

    # integrate the neural network outputs over a sliding window
    integrated = integrator(y, window_size)

    if normalizer is not None:
        normalizer.fit(y)
        integrated = normalizer(integrated, window_size)
        y = y[-len(integrated) :]
        t = t[-len(integrated) :]

    # offset timesteps by kernel size so that they
    # refer to the time at the front of the kernel
    # rather than the back for comparison to trigger times
    return t + kernel_length, y, integrated
