from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from bbhnet.analysis.integrators import boxcar_filter

if TYPE_CHECKING:
    from bbhnet.analysis.integrators import Integrator
    from bbhnet.analysis.normalizers import Normalizer


def integrate(
    y: np.ndarray,
    t: np.ndarray,
    kernel_length: float = 1,
    window_length: Optional[float] = None,
    integrator: "Integrator" = boxcar_filter,
    normalizer: Optional["Normalizer"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze a segment of time-contiguous BBHNet outputs

    Compute matched filter outputs on an array of
    network outputs that are assumed to be contiguous
    and ordered in time. Matched filters are computed
    as the average over the the last `window_length`
    seconds of data. Optionally normalized by the mean
    and standard deviation of the previous `norm_seconds`
    seconds worth of data.

    Args:
        y: Array of network outputs to integrate
        t: timestamps of network outputs
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
            A pre-fit Normalizer object

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
    sample_rate = 1 / (t[1] - t[0])
    window_length = window_length or kernel_length
    window_size = int(window_length * sample_rate)

    # integrate the neural network outputs over a sliding window
    integrated = integrator(y, window_size)

    if normalizer is not None:
        integrated = normalizer(integrated, window_size)
        y = y[-len(integrated) :]
        t = t[-len(integrated) :]

    return t, y, integrated
