import numpy as np
import pytest

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.integrators import boxcar_filter
from bbhnet.analysis.normalizers import GaussianNormalizer


@pytest.fixture(params=[101, 128, 256])
def length(request):
    return request.param


@pytest.fixture(params=[8, 16])
def sample_rate(request):
    return request.param


@pytest.fixture
def y(length, sample_rate):
    return np.arange(int(length * sample_rate))


@pytest.fixture
def t(length, sample_rate):
    return np.arange(0, length, 1 / sample_rate)


@pytest.fixture(params=[1, 1.5, 2])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[None, 1])
def window_length(request):
    return request.param


@pytest.fixture(params=[None, 10, 15])
def normalizer(request, sample_rate):
    if request.param is None:
        return None
    return GaussianNormalizer(int(request.param * sample_rate))


def test_integrate(
    y,
    t,
    kernel_length,
    window_length,
    normalizer,
    sample_rate,
    boxcar_integration_test_fn,
):
    assert len(y) == len(t)

    if normalizer is not None:
        normalizer.fit(y)

    t_, y_, integrated = integrate(
        y, t, kernel_length, window_length, boxcar_filter, normalizer
    )

    window_length = window_length or kernel_length
    expected_length = len(y)
    if normalizer is not None:
        expected_length -= (
            int(window_length * sample_rate) + normalizer.norm_size
        )
    assert len(t_) == len(y_) == len(integrated) == expected_length

    assert (t_ == (t[-len(t_) :])).all()
    assert (y_ == (y[-len(y_) :])).all()

    window_size = window_length * sample_rate
    if normalizer is None:
        boxcar_integration_test_fn(window_size, integrated)
    else:
        # TODO: work out the math for the expected
        # result if a normalizer was used. Should be
        # reasonably straightforward
        pass
