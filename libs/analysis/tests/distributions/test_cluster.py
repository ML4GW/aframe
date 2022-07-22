from unittest.mock import MagicMock, Mock

import numpy as np

from bbhnet.analysis.distributions.cluster import ClusterDistribution
from bbhnet.io.timeslides import Segment


def test_cluster_distribution():
    distribution = ClusterDistribution("test", 5)

    y = t = np.arange(20).astype("float32")
    distribution.update(y, t)
    assert distribution.Tb == 20
    assert distribution.events == [4.0, 9.0, 14.0, 19.0]

    y += 1
    distribution.update(y, t)
    assert distribution.Tb == 40
    assert distribution.events == [
        4.0,
        9.0,
        14.0,
        19.0,
        5.0,
        10.0,
        15.0,
        20.0,
    ]

    segment = Mock(Segment)
    segment.load = MagicMock(return_value=(y, t))
    segment.fnames = ["test1"]

    distribution.fit(segment)
    assert distribution.Tb == 60
    assert distribution.events == [
        4.0,
        9.0,
        14.0,
        19.0,
        5.0,
        10.0,
        15.0,
        20.0,
        5.0,
        10.0,
        15.0,
        20.0,
    ]

    # test situation where data length is not
    # divisible by t_clust
    y = t = np.arange(19).astype("float32")
    segment = Mock(Segment)
    segment.load = MagicMock(return_value=(y, t))
    segment.fnames = ["test1"]
    distribution.fit(segment, warm_start=False)
    assert distribution.Tb == 19
    assert distribution.events == [4.0, 9.0, 14.0, 18.0]

    assert distribution.nb(1) == 4
    assert distribution.nb(10) == 2
