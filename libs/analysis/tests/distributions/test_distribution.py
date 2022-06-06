from copy import deepcopy
from math import exp
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from bbhnet.analysis.distributions.distribution import (
    SECONDS_IN_YEAR,
    Distribution,
)
from bbhnet.io.timeslides import Segment


def nb(x):
    try:
        return np.array([10 for _ in x])
    except TypeError:
        return 10


@pytest.fixture
def sample_rate():
    return 4


@pytest.fixture(params=[0, 1, -1])
def offset(request, sample_rate):
    return request.param / (2 * sample_rate)


@pytest.fixture
def event_time(offset):
    return SECONDS_IN_YEAR / 100 - offset


@pytest.fixture(params=[float, list])
def event_times(request, event_time, sample_rate):
    if request.param == float:
        return event_time
    else:
        return np.array([event_time + i * 10 for i in range(3)])


def test_distribution(event_time, event_times, offset, sample_rate):
    distribution = Distribution("test")
    distribution.nb = nb
    distribution.Tb = SECONDS_IN_YEAR / 2

    # unit test characterization methods
    # false alarm rate
    far = distribution.far(0)
    assert far == 20

    # significance
    tf = (event_time + offset) * 2
    significance = distribution.significance(0, tf)
    assert significance == 1 - exp(-tf * 22 / SECONDS_IN_YEAR)

    # now create some dummy event segments and ensure
    # that the characterization curves that come out
    # of them match up
    t = np.arange(0, tf, 1 / sample_rate)
    y = np.ones_like(t)
    segment = Mock(Segment)
    segment.load = MagicMock(return_value=(y, t))

    for metric, expected in zip(["far", "significance"], [far, significance]):
        characterization, times = distribution.characterize_events(
            segment, event_times=event_times, window_length=1, metric=metric
        )

        # make sure the appropriate segment data got "loaded"
        segment.load.assert_called_with("test")

        # for a single event, make sure we have 1D
        # timeseries of the appropriate length and content
        if isinstance(event_times, float):
            assert characterization.ndim == 1
            assert len(characterization) == sample_rate

            assert times.ndim == 1
            start = int(len(t) // 2) + 1
            if offset > 0:
                start -= 1

            t_expect = t[start : start + sample_rate] - event_time
            assert np.isclose(times, t_expect, rtol=1e-9).all()
        else:
            # for 2D, make sure we have 3 events with the right
            # length and content
            assert characterization.shape == (3, sample_rate)
            assert times.shape == (3, sample_rate)

            assert times.ndim == 2
            assert len(times) == 3
            assert times.shape[-1] == sample_rate

            for i, tc in enumerate(event_times):
                start = int(len(t) // 2) + 1
                if offset > 0:
                    start -= 1
                start += i * 10 * sample_rate

                t_expect = t[start : start + sample_rate] - tc
                assert np.isclose(times[i], t_expect, rtol=1e-9).all()

        # characterization values will be constant since
        # that's how we've set up `nb` to return things
        assert np.isclose(characterization, expected, rtol=1e-7).all()

    # override the update method to make sure
    # it gets called appropriately and the right
    # number of times TODO: a MagicMock could
    # probably do this just as well
    def update(*args):
        distribution.Tb += 10

    distribution.update = update

    # test fitting on a single segment
    segment.fnames = ["test1"]
    distribution.fit(segment)

    # fnames are right, backround time updated
    assert distribution.fnames == ["test1"]
    assert distribution.Tb == SECONDS_IN_YEAR / 2 + 10

    # now create a list of segments to fit on
    segment2 = deepcopy(segment)
    segment.fnames = ["test2", "test3"]
    segment2.fnames = ["test4"]
    distribution.fit([segment, segment2])

    # same checks
    assert distribution.fnames == [f"test{i + 1}" for i in range(4)]
    assert distribution.Tb == SECONDS_IN_YEAR / 2 + 30

    # now fit on the list but without warm starting,
    # which should reset `fnames` and `Tb`
    distribution.fit([segment, segment2], warm_start=False)
    assert distribution.fnames == [f"test{i + 2}" for i in range(3)]
    assert distribution.Tb == 20
