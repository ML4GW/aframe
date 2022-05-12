import numpy as np
from generate_glitches import veto
from gwpy.segments import Segment, SegmentList

# some very non-exhaustive test of veto script
# don't think we're using vetos at this moment anyway
# will likely move this to some future 'post-production analysis' library


def test_veto_all_triggers():

    # arange sample trigger times
    trig_times = np.arange(0, 100, 1)

    # create vetoes that cover all time
    vetoes = SegmentList([Segment(-np.inf, np.inf)])

    keep_bools = veto(trig_times, vetoes)

    trig_times_post_veto = trig_times[keep_bools]

    # assert that no trigger times are kept
    assert len(trig_times_post_veto) == 0


def test_veto_no_triggers():

    # arange sample trigger times
    trig_times = np.arange(0, 100, 1)

    # create vetoes that dont cover any of trigger times
    vetoes = SegmentList([Segment(-100, -10), Segment(200, 300)])

    keep_bools = veto(trig_times, vetoes)

    trig_times_post_veto = trig_times[keep_bools]

    # assert that no trigger times are kept
    assert len(trig_times_post_veto) == len(trig_times)


def test_veto_some_triggers():

    trig_times = np.array([5, 12, 25, 32, 41])
    vetoes = SegmentList([Segment(0, 10), Segment(20, 30), Segment(30, 40)])

    keep_bools = veto(trig_times, vetoes)
    trig_times_post_veto = trig_times[keep_bools]

    assert 5 not in trig_times_post_veto
    assert 12 in trig_times_post_veto
    assert 32 not in trig_times_post_veto
