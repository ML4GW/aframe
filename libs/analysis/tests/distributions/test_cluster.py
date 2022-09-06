from unittest.mock import MagicMock, Mock

import numpy as np

from bbhnet.analysis.distributions.cluster import ClusterDistribution
from bbhnet.io.timeslides import Segment


def test_cluster_distribution():

    t_clust = 2
    distribution = ClusterDistribution("test", ["H", "L"], t_clust)

    # expect the first and third element
    # to cluster with the middle
    t = np.array([1, 2, 3])
    y = np.array([1, 2, 1])

    shifts = [0, 1]

    distribution.fit((y, t), shifts)
    assert distribution.Tb == 3
    assert (distribution.events == [2]).all()
    assert len(distribution.shifts) == len(distribution.events)

    # expect no clustering
    t = np.array([0, 2, 4])
    y = np.array([1, 2, 1])

    distribution.fit((y, t), shifts)
    assert distribution.Tb == 9
    assert (distribution.events == [2, 1, 2, 1]).all()
    assert len(distribution.shifts) == len(distribution.events)

    # expect all 1s to cluster to 2s
    t = np.array([1, 2, 3, 4, 5, 6, 7])
    y = np.array([1, 2, 1, 2, 1, 2, 1])

    distribution.fit((y, t), shifts)
    assert distribution.Tb == 16
    assert (distribution.events == [2, 1, 2, 1, 2, 2, 2]).all()
    assert len(distribution.shifts) == len(distribution.events)

    assert distribution.nb(0) == 7
    assert distribution.nb(1) == 7
    assert distribution.nb(2) == 5

    # if sample rate larger than t_clust,
    # all events should stay
    t = np.array([8, 16, 24, 32])
    y = np.array([1, 2, 3, 4])
    distribution.fit((y, t), shifts)
    assert (distribution.events == [2, 1, 2, 1, 2, 2, 2, 1, 2, 3, 4]).all()

    # TODO: do we want the below behavior?

    # if the network output trends upward
    # on time scale larger than t_clust,
    # all of the events will cluster to the maximium.
    # i.e only one event will remain in this 9 second
    # window, even though our clustering full window is
    # 2 seconds. As long as our clustering time scale
    # is longer than the time scale over which
    # we expect our network output to increase (i.e. our kernel length)
    # this shouldnt be a problem, but something to note.

    t_clust = 2
    distribution = ClusterDistribution("test", ["H", "L"], t_clust)

    t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    distribution.fit((y, t), shifts)

    assert (distribution.events == [9]).all()

    # now test fitting with segments
    # from scratch
    t = np.array([1, 2, 3])
    y = np.array([1, 2, 1])

    shifts = [0, 1]
    segment = Mock(Segment)
    segment.load = MagicMock(return_value=(y, t))
    segment.shifts = shifts

    distribution.fit(segment, warm_start=False)
    assert distribution.Tb == 3
    assert (distribution.events == [2]).all()
    assert len(distribution.shifts) == len(distribution.events)

    # list of segments
    distribution.fit([segment, segment], warm_start=False)
    assert distribution.Tb == 6
    assert (distribution.events == [2, 2]).all()
    assert len(distribution.shifts) == len(distribution.events)
