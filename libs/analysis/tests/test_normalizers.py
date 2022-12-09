from math import isclose

import numpy as np
import pytest

from bbhnet.analysis.normalizers import GaussianNormalizer


@pytest.fixture(
    params=[
        16,
        17,
        160,
        16.0,
        pytest.param(16.2, marks=pytest.mark.xfail(raises=ValueError)),
    ]
)
def norm_size(request):
    return request.param


def test_gaussian_normalizer(
    norm_size, window_size, boxcar_integration_test_fn
):
    normalizer = GaussianNormalizer(norm_size)

    # silly check but doing it solely
    # for the sake of checking int-ification
    assert normalizer.norm_size == int(norm_size)
    norm_size = int(norm_size)

    # test to make sure we enforce fitting
    y = np.arange(1000)
    with pytest.raises(ValueError) as exc_info:
        normalizer(y, window_size)
    assert str(exc_info.value) == "GaussianNormalizer hasn't been fit"

    # test to make sure we don't allow 0 scale values
    with pytest.raises(ValueError) as exc_info:
        normalizer.fit(y)
    assert str(exc_info.value) == "Encountered 0s in scale parameter"

    normalizer.fit(y + 1)
    boxcar_integration_test_fn(norm_size, normalizer.shifts)

    # check the scale values manually
    for i, value in enumerate(normalizer.scales):
        value = value**2
        if i < norm_size:
            mu = normalizer.shifts[i]
            expected = (norm_size - i - 1) * mu**2
            expected += sum([(j + 1 - mu) ** 2 for j in range(i + 1)])
            expected /= norm_size
        else:
            expected = (norm_size**2 - 1) / 12
        assert isclose(value, expected, rel_tol=1e-9), i

    with pytest.raises(ValueError) as exc_info:
        normalizer(y[:-1], window_size)
    assert str(exc_info.value).startswith("Can't normalize")

    normalized = normalizer(y + 1, window_size)
    assert len(normalized) == (len(y) - window_size - norm_size)

    # all the normalized values should be equal to a constant
    expected = (3 / (norm_size**2 - 1)) ** 0.5
    expected *= norm_size + 2 * window_size - 1
    assert np.isclose(expected, normalized, rtol=1e-9).all()
