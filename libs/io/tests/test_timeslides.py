import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide, filter_and_sort_files


@pytest.fixture
def t0():
    return 1234567890


@pytest.fixture
def shift_dir():
    return "dt-H0.0-L4.0"


@pytest.fixture
def field():
    return "nn"


@pytest.fixture
def timeslide_dir(tmpdir, shift_dir, field):
    timeslide_dir = tmpdir / shift_dir / field
    timeslide_dir.mkdir(parents=True, exist_ok=False)
    return timeslide_dir


@pytest.fixture(params=[1024, 4096])
def file_length(request):
    return request.param


@pytest.fixture
def sample_rate():
    return 4


@pytest.fixture
def segment_fnames(timeslide_dir, t0, file_length, sample_rate):
    fnames = []
    num_samples = sample_rate * file_length
    for i in range(3):
        start = t0 + i * file_length
        t = np.arange(start, start + file_length, 1 / sample_rate)
        y = np.arange(i * num_samples, (i + 1) * num_samples)
        other = -y
        fname = write_timeseries(timeslide_dir, t=t, y=y, other=other)
        fnames.append(fname)
    return fnames


@pytest.fixture(params=[str, Path])
def path_type(request):
    return request.param


# params mean:
# 0: return as individual `path_type`
# 1: return len(1) list
# None: return filenames as-is
# -1: reverse filename list to test segment ordering
@pytest.fixture(params=[0, 1, None, -1])
def typed_segment_fnames(segment_fnames, path_type, request):
    if request.param is None or request.param == -1:
        # map segment filenames to the specified type
        fnames = list(map(path_type, segment_fnames))

        # if -1 reorder things to make sure Segment can
        # handle ordering them itself
        if request.param == -1:
            fnames = fnames[::-1]
        return fnames
    elif request.param == 0:
        # return a standalone PATH_LIKE object
        return path_type(segment_fnames[0])
    elif request.param == 1:
        # return a single filename in a list
        return [path_type(segment_fnames[0])]


def test_filter_and_sort_files(
    typed_segment_fnames, path_type, segment_fnames
):
    if isinstance(typed_segment_fnames, (str, Path)):
        typed_segment_fnames = Path(typed_segment_fnames).parent
        typed_segment_fnames = path_type(typed_segment_fnames)
        expected_fnames = segment_fnames
        expected_type = Path
    else:
        expected_fnames = segment_fnames[: len(typed_segment_fnames)]
        expected_fnames = list(map(path_type, expected_fnames))
        expected_type = path_type

    result = filter_and_sort_files(typed_segment_fnames)

    assert len(result) == len(expected_fnames)
    assert all([isinstance(i, expected_type) for i in result])
    assert all([i == j for i, j in zip(result, expected_fnames)])

    expected_fnames = [Path(i).name for i in expected_fnames]
    matches = filter_and_sort_files(typed_segment_fnames, return_matches=True)
    assert len(result) == len(expected_fnames)
    assert all([i.string == j for i, j in zip(matches, expected_fnames)])


def test_segment_basics(
    tmpdir, shift_dir, field, typed_segment_fnames, t0, file_length
):
    segment = Segment(typed_segment_fnames)

    # test that all the properties got initialized correctly
    if isinstance(typed_segment_fnames, (str, Path)):
        num_files = 1
    else:
        num_files = len(typed_segment_fnames)

    assert segment.t0 == t0
    assert segment.length == (file_length * num_files)
    assert segment.tf == (t0 + file_length * num_files)
    # make sure the __contains__ method works right
    for t in range(int(t0), int(t0 + segment.length)):
        assert t in segment

    # make sure these properties check out
    assert segment.root == tmpdir / shift_dir / field
    assert segment.field == field
    assert segment.shift_dir == shift_dir
    assert segment.shifts == [0.0, 4.0]
    assert segment.ifos == ["H", "L"]


def test_segment_init_errors(tmpdir, shift_dir, field, segment_fnames):
    # make sure a nonexistent file raises a ValueError
    new_fname = tmpdir / "dt-1.0" / field / segment_fnames[0].name
    with pytest.raises(ValueError) as exc_info:
        Segment([new_fname] + segment_fnames[1:])
    assert str(exc_info.value) == f"Segment file {new_fname} does not exist"

    # make sure files from different directories
    # raise a different ValueError. Ensure the
    # file exists first so that the previous
    # error doesn't get tripped first
    new_fname.parent.mkdir(parents=True)
    with open(new_fname, "w"):
        pass

    with pytest.raises(ValueError) as exc_info:
        Segment([new_fname] + segment_fnames[1:])
    assert str(exc_info.value).startswith("Segment files")


def test_segment_loading(segment_fnames, t0, file_length, sample_rate):
    # create a segment and load its data
    segment = Segment(segment_fnames)
    y, t = segment.load("out")

    # verify that the data matches our expectations
    expected_length = sample_rate * file_length * 3
    assert len(y) == len(t) == expected_length
    assert (y == np.arange(expected_length)).all()
    assert (t == np.arange(t0, t0 + file_length * 3, 1 / sample_rate)).all()

    # make sure cache elements are set correctly
    assert "out" in segment._cache
    assert (segment._cache["out"] == y).all()

    assert "t" in segment._cache
    assert (segment._cache["t"] == t).all()

    # now patch the read method to make sure that when
    # we call load again, it doesn't attempt to read but
    # goes to the cache instead
    with patch("bbhnet.io.timeslides.Segment.read") as mock:
        y, t = segment.load("out")
        mock.assert_not_called()

        # make sure that the values loaded from the cache are correct
        assert len(y) == len(t) == expected_length
        assert (y == np.arange(expected_length)).all()
        assert (
            t == np.arange(t0, t0 + file_length * 3, 1 / sample_rate)
        ).all()

        # loading another field though should need
        # to make a call to read_timeseries. Give
        # our mock a return value to make things
        # match size-wise
        file_size = sample_rate * file_length
        mock.return_value = y[:file_size], t[:file_size]

        other, t_cached = segment.load("other")
        mock.assert_called_with(segment_fnames[-1], "other")

        # t should have been read from cache
        assert (t_cached == t).all()

        # other should just look like the first
        # file out y repeated 3 times
        expected = np.concatenate([y[:file_size]] * 3)
        assert (other == expected).all()

    # make sure that this other field also gets loaded right
    segment._cache.pop("other")
    other, t = segment.load("other")
    assert (other == -y).all()


def test_append(
    tmpdir, shift_dir, field, t0, file_length, path_type, segment_fnames
):
    segment = Segment(segment_fnames[:-1])
    assert segment.length == (file_length * (len(segment_fnames) - 1))

    # load something to verify the cache gets reset
    segment.load("out")
    assert len(segment._cache) == 2

    # TODO: test re.Match behavior as well
    segment.append(path_type(segment_fnames[-1]))
    assert segment.length == (file_length * len(segment_fnames))
    assert len(segment._cache) == 0
    assert segment.fnames[-1] == segment_fnames[-1]

    # reinitialize the segment and test some of the errors
    segment = Segment(segment_fnames[:-1])

    # test to make sure a mismatched root causes a problem
    bad_fname = tmpdir / shift_dir / "bad" / segment_fnames[-1].name
    with pytest.raises(ValueError) as exc_info:
        segment.append(path_type(bad_fname))
    assert str(exc_info.value).startswith(
        f"Can't append filename '{bad_fname}'"
    )

    # poorly formatted filenames won't work
    t = t0 + (len(segment_fnames) - 1) * file_length
    bad_fname = segment_fnames[-1].parent / f"out_{t}--{file_length}.hdf5"
    with pytest.raises(ValueError) as exc_info:
        segment.append(path_type(bad_fname))
    assert str(exc_info.value).startswith(
        f"Filename '{bad_fname}' not properly formatted"
    )

    # non-contiguous files won't work
    t_bad = t + 1
    bad_fname = segment_fnames[-1].parent / f"out_{t_bad}-{file_length}.hdf5"
    with pytest.raises(ValueError) as exc_info:
        segment.append(path_type(bad_fname))
    assert str(exc_info.value).startswith(
        f"Filename '{bad_fname}' has timestamp {t_bad}"
    )


def test_segment_make_shift(tmpdir, shift_dir, field, segment_fnames):
    new_dir = tmpdir / "dt-H1.0-L2.0" / field
    new_dir.mkdir(parents=True)
    for fname in segment_fnames:
        shutil.copy(fname, new_dir / fname.name)

    segment = Segment(segment_fnames)
    new_segment = segment.make_shift("dt-H1.0-L2.0")
    assert new_segment.root == new_dir
    assert new_segment.shift_dir == "dt-H1.0-L2.0"
    assert new_segment.field == segment.field
    assert new_segment.t0 == segment.t0
    assert new_segment.length == segment.length

    with pytest.raises(ValueError):
        segment.make_shift("dt-H0.0-L5.0")


@pytest.fixture
def more_segment_fnames(timeslide_dir, t0, file_length, sample_rate):
    fnames = []
    num_samples = sample_rate * file_length
    for i in [3, 1, 4]:
        segment_fnames = []
        for j in range(i):
            start = t0 + j * file_length
            t = np.arange(start, start + file_length, 1 / sample_rate)
            y = np.arange(i * num_samples, (i + 1) * num_samples)
            other = -y
            fname = write_timeseries(timeslide_dir, t=t, y=y, other=other)
            segment_fnames.append(fname)

        t0 = t0 + (i + 1) * file_length
        fnames.append(segment_fnames)
    return fnames


def test_timeslide(tmpdir, shift_dir, more_segment_fnames):
    ts = TimeSlide(tmpdir / shift_dir, "nn")
    assert ts.path == tmpdir / shift_dir / "nn"
    assert ts.shift_dir == shift_dir

    assert len(ts.segments) == len(more_segment_fnames)
    for segment, fnames in zip(ts.segments, more_segment_fnames):
        assert len(segment.fnames) == len(fnames)

        for segment_f, f in zip(segment.fnames, fnames):
            assert segment_f == f


def test_create_timeslide(tmpdir, shift_dir, more_segment_fnames):
    ts = TimeSlide.create(tmpdir / shift_dir, "nn")
    assert ts.path == tmpdir / shift_dir / "nn"
    assert ts.shift_dir == shift_dir

    assert len(ts.segments) == len(more_segment_fnames)
    for segment, fnames in zip(ts.segments, more_segment_fnames):
        assert len(segment.fnames) == len(fnames)

        for segment_f, f in zip(segment.fnames, fnames):
            assert segment_f == f


# TODO: test TimeSlide.create() when path doesnt already exist
