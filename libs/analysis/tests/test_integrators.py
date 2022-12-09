import numpy as np

from bbhnet.analysis.integrators import boxcar_filter


def test_boxcar_filter(window_size, boxcar_integration_test_fn):
    y = np.arange(1000) + 1
    result = boxcar_filter(y, window_size)
    assert len(result) == len(y)

    boxcar_integration_test_fn(window_size, result)
