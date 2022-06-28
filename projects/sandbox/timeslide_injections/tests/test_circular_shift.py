import os
import shutil
from pathlib import Path

import pytest
from gwpy.segments import Segment, SegmentList
from timeslide_injections import circular_shift_segments


@pytest.fixture(scope="session")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[-1, 0, 1])
def shift(request):
    return request.param


def test_circular_shift_segments_with_zero_shift():
    start = 0
    stop = 1000
    shift = 0
    segmentlist = SegmentList(
        [Segment([0, 100]), Segment([200, 300]), Segment([400, 1000])]
    )

    circularly_shifted = circular_shift_segments(
        segmentlist, shift, start, stop
    )

    assert circularly_shifted == segmentlist


def test_circular_shift_segments_with_full_segment():
    start = 0
    stop = 1000
    shift = 0
    segmentlist = SegmentList([Segment([start, stop])])

    circularly_shifted = circular_shift_segments(
        segmentlist, shift, start, stop
    )

    assert circularly_shifted == segmentlist


def test_circular_shift_segments_with_positive_shift():
    start = 0
    stop = 600
    shift = 100
    segmentlist = SegmentList(
        [Segment([0, 100]), Segment([200, 300]), Segment([400, 550])]
    )
    expected_output = SegmentList(
        [
            Segment([0, 50]),
            Segment([100, 200]),
            Segment([300, 400]),
            Segment([500, 600]),
        ]
    )

    circularly_shifted = circular_shift_segments(
        segmentlist,
        shift,
        start,
        stop,
    )

    assert circularly_shifted == expected_output
