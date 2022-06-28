import shutil
from pathlib import Path

import numpy as np
import pytest
from gwpy.segments import Segment, SegmentList

from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide


@pytest.fixture(scope="function")
def tmpdir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def shift():
    return "dt-1.0"


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture
def timeslide_dir(tmpdir, shift):
    timeslide_dir = tmpdir / shift
    timeslide_dir.mkdir(parents=True, exist_ok=False)
    return timeslide_dir


@pytest.fixture(params=[1024, 4096])
def file_length(request):
    return request.param


@pytest.fixture
def segments():
    return SegmentList(
        [
            Segment([1256600000, 1256604096]),
            Segment([1256608000, 1256609000]),
        ]
    )


@pytest.fixture
def make_segments(timeslide_dir, segments, sample_rate, file_length, ifos):
    fnames = []
    for segment in segments:
        start, stop = segment
        segment_fnames = []
        for t0 in np.arange(start, stop, file_length):

            tf = min(t0 + file_length, stop)

            t = np.arange(t0, tf, 1 / sample_rate)
            datasets = {}
            for ifo in ifos:
                datasets[ifo] = np.zeros(len(t))

            field_dir = timeslide_dir / "raw"
            field_dir.mkdir(exist_ok=True, parents=True)

            fname = h5.write_timeseries(
                field_dir, prefix="raw", t=t, **datasets
            )
            segment_fnames.append(fname)

        fnames.append(segment_fnames)

    return fnames


@pytest.fixture
def raw_timeslide(timeslide_dir, make_segments):
    field_dir = timeslide_dir / "raw"
    field_dir.mkdir(exist_ok=True, parents=True)
    return TimeSlide(timeslide_dir, field="raw")


@pytest.fixture
def inj_timeslide(timeslide_dir, make_segments):
    field_dir = timeslide_dir / "inj"
    field_dir.mkdir(exist_ok=True, parents=True)
    return TimeSlide(timeslide_dir, field="inj")
