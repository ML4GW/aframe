from typing import Callable

import numpy as np
from scipy.signal import convolve

Integrator = Callable[[np.ndarray, int], np.ndarray]


def boxcar_filter(y: np.ndarray, window_size: int) -> np.ndarray:
    window = np.ones((window_size,)) / window_size
    mf = convolve(y, window, mode="full")
    return mf[: -window_size + 1]
