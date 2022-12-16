from math import isclose

import pytest


@pytest.fixture(params=[16, 17, 128])
def window_size(request):
    return request.param


@pytest.fixture
def boxcar_integration_test_fn():
    def _test_fn(window_size, array):
        for i, value in enumerate(array):
            if i < window_size:
                expected = sum(range(i + 2)) / window_size
            else:
                expected = (i + 3 + i - window_size) / 2
            assert isclose(value, expected, rel_tol=1e-9)

    return _test_fn
