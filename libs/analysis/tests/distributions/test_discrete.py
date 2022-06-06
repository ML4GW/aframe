from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from bbhnet.analysis.distributions.discrete import DiscreteDistribution
from bbhnet.io.timeslides import Segment


@pytest.fixture(params=[True, False])
def clip(request):
    return request.param


def test_discrete_distribution(clip):
    distribution = DiscreteDistribution("test", 0, 10, 10, clip=clip)
    assert len(distribution.bins) == 11
    assert len(distribution.histogram) == 10

    y = t = np.arange(10).astype("float32")
    distribution.update(y, t)
    assert (distribution.histogram == 1).all()
    assert distribution.Tb == 10

    y += 0.5
    distribution.update(y, t)
    assert (distribution.histogram == 2).all()
    assert distribution.Tb == 20

    extra = 0 if clip else 1
    y += 1
    distribution.update(y, t)
    assert distribution.histogram[0] == 2
    assert (distribution.histogram[1:-1] == 3).all()
    assert distribution.histogram[-1] == 3 + extra
    assert distribution.Tb == 30

    segment = Mock(Segment)
    segment.load = MagicMock(return_value=(y, t))
    segment.fnames = ["test1"]

    distribution.fit(segment)
    assert distribution.histogram[0] == 2
    assert (distribution.histogram[1:-1] == 4).all()
    assert distribution.histogram[-1] == 4 + 2 * extra
    assert distribution.Tb == 40

    distribution.fit([segment, segment])
    assert distribution.histogram[0] == 2
    assert (distribution.histogram[1:-1] == 6).all()
    assert distribution.histogram[-1] == 6 + 4 * extra
    assert distribution.Tb == 60

    distribution.fit(segment, warm_start=False)
    assert distribution.histogram[0] == 0
    assert (distribution.histogram[1:-1] == 1).all()
    assert distribution.histogram[-1] == 1 + extra
    assert distribution.Tb == 10

    assert distribution.nb(5) == 5 + extra
    assert distribution.nb(0.5) == 9 + extra

    assert (
        distribution.nb(np.array([0.5, 5])) == np.array([9, 5]) + extra
    ).all()

    # TODO: test write/read/load
