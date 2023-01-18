from unittest.mock import Mock, patch

import numpy as np
import pytest
from datagen.utils.priors import read_priors_from_file
from scipy.stats import ks_2samp


@pytest.fixture
def event_params():
    event_dtype = np.dtype(
        [
            ("mass_1", "<f8"),
            ("mass_ratio", "<f8"),
            ("a_1", "<f8"),
            ("a_2", "<f8"),
            ("cos_tilt_1", "<f8"),
            ("cos_tilt_2", "<f8"),
            ("redshift", "<f8"),
            ("mass_2", "<f8"),
        ]
    )
    num_events = 10000
    rand = np.random.rand(num_events)
    params = np.array(8 * np.array(rand), dtype=event_dtype)

    return params


@pytest.fixture
def h5py_mock(event_params):
    def mock(fname, _):
        value = {"events": event_params}
        obj = Mock()
        obj.__enter__ = lambda obj: value
        obj.__exit__ = Mock()
        return obj

    with patch("h5py.File", new=mock):
        yield mock


def test_pdf_from_events(h5py_mock, event_params):
    prior = read_priors_from_file(h5py_mock)
    interp_sampled = prior.sample(10000)
    for name in event_params.dtype.names:
        _, pval = ks_2samp(interp_sampled[name], event_params[name])
        assert pval > 0.05
