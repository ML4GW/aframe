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
    distribution = Distribution("test", ["H", "L"])
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

    # override the update method to make sure
    # it gets called appropriately and the right
    # number of times TODO: a MagicMock could
    # probably do this just as well
    def update(*args):
        distribution.Tb += 10

    distribution.update = update

    # test fitting on a single segment
    distribution.fit(segment)

    # fnames are right, backround time updated
    assert distribution.Tb == SECONDS_IN_YEAR / 2 + 10

    # now create a list of segments to fit on
    segment2 = deepcopy(segment)
    distribution.fit([segment, segment2])

    # same checks
    assert distribution.Tb == SECONDS_IN_YEAR / 2 + 30

    # now fit on the list but without warm starting,
    # which should reset `fnames` and `Tb`
    distribution.fit([segment, segment2], warm_start=False)
    assert distribution.Tb == 20

    # now test fitting by passing a tuple
    distribution.fit((y, t))
    assert distribution.Tb == 30

    # test apply vetoes function
    distribution = Distribution("test", ["H", "L"])
    distribution.events = np.arange(10)
    distribution.event_times = np.arange(10)

    # first test with no shifts
    shifts = [0, 0]
    distribution.shifts = np.repeat([shifts], len(distribution.events), axis=0)

    # should vetoe events 3,4,5,6,7
    L_vetoes = [[2, 8]]
    H_vetoes = [[2, 8]]
    distribution.apply_vetoes(L=L_vetoes)
    assert (distribution.event_times == [0, 1, 2, 8, 9]).all()
    assert (distribution.events == [0, 1, 2, 8, 9]).all()
    assert len(distribution.shifts) == 5

    # now reapply vetoes on H;
    # shouldn't vetoe any events
    distribution.apply_vetoes(H=H_vetoes)
    assert (distribution.event_times == [0, 1, 2, 8, 9]).all()
    assert (distribution.events == [0, 1, 2, 8, 9]).all()
    assert len(distribution.shifts) == 5

    # now test with shifts
    distribution = Distribution("test", ["H", "L"])
    distribution.events = np.arange(20)
    distribution.event_times = np.arange(20)
    shifts = [0, 20]
    distribution.shifts = np.repeat([shifts], len(distribution.events), axis=0)

    # first five events
    H_vetoes = [[-1, 4.5], [4.6, 5]]

    # last five events
    L_vetoes = [[-6, -2.9], [-2.8, 0]]

    expected_events = np.arange(5, 15, 1)

    distribution.apply_vetoes(H=H_vetoes)
    distribution.apply_vetoes(L=L_vetoes)

    print(distribution.event_times)
    assert (distribution.event_times == expected_events).all()
    assert (distribution.events == expected_events).all()
    assert len(distribution.shifts) == 10

    # re applying vetoes shouldn't change anything
    distribution.apply_vetoes(H=H_vetoes)
    distribution.apply_vetoes(L=L_vetoes)

    assert (distribution.event_times == expected_events).all()
    assert (distribution.events == expected_events).all()
    assert len(distribution.shifts) == 10
