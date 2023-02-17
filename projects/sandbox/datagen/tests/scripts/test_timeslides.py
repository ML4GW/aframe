from concurrent.futures._base import FINISHED
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.cosmology import Planck15
from datagen.scripts.timeslides import main as generate_timeslides
from gwpy.segments import (
    DataQualityDict,
    DataQualityFlag,
    Segment,
    SegmentList,
)
from gwpy.timeseries import TimeSeries

from bbhnet.io.timeslides import TimeSlide
from bbhnet.priors.priors import end_o3_ratesandpops


@pytest.fixture(params=[end_o3_ratesandpops])
def prior(request):
    return request.param


@pytest.fixture(params=[30, 60])
def spacing(request):
    return request.param


@pytest.fixture(params=[2])
def jitter(request):
    return request.param


@pytest.fixture(params=[0, 10])
def buffer_(request):
    return request.param


@pytest.fixture(params=[1, 5, 10])
def n_slides(request):
    return request.param


@pytest.fixture(params=[32])
def highpass(request):
    return request.param


@pytest.fixture(params=[20])
def minimum_frequency(request):
    return request.param


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[[0, 1]])
def shifts(request):
    return request.param


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=["HOFT_C01"])
def frame_type(request):
    return request.param


@pytest.fixture(params=["DCS-CALIB_STRAIN_CLEAN_C01"])
def channel(request):
    return request.param


@pytest.fixture(params=["DCS-ANALYSIS_READY_C01:1"])
def state_flag(request):
    return request.param


@pytest.fixture()
def cosmology():
    def f():
        return Planck15

    return f


def submit_mock(f, *args, **kwargs):
    result = Mock()
    value = f(*args, **kwargs)
    result.result = Mock(return_value=value)
    result._state = FINISHED
    return result


def mock_exit(self, *exc_args):
    # return False to process exceptions!
    return False


pool_mock = Mock()
pool_mock.submit = submit_mock


@patch("bbhnet.parallelize.AsyncExecutor.__enter__", return_value=pool_mock)
@patch("bbhnet.parallelize.AsyncExecutor.__exit__", new=mock_exit)
def test_timeslide_injections_no_segments(
    mock1,
    logdir,
    datadir,
    prior,
    spacing,
    jitter,
    buffer_,
    n_slides,
    shifts,
    ifos,
    minimum_frequency,
    highpass,
    sample_rate,
    frame_type,
    channel,
    cosmology,
):
    start = 1123456789
    stop = 1123457789

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    x = np.arange(n_samples).astype("float64")
    ts = TimeSeries(x, times=times)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)
    mock_datafind = patch("gwdatafind.find_urls", return_value=None)
    with mock_datafind, mock_ts:
        generate_timeslides(
            start=start,
            stop=stop,
            logdir=logdir,
            datadir=datadir,
            prior=prior,
            spacing=spacing,
            jitter=jitter,
            buffer_=buffer_,
            n_slides=n_slides,
            shifts=shifts,
            ifos=ifos,
            minimum_frequency=minimum_frequency,
            highpass=highpass,
            sample_rate=sample_rate,
            frame_type=frame_type,
            channel=channel,
            cosmology=cosmology,
        )

    timeslides = datadir.iterdir()
    timeslides = [slide for slide in timeslides if slide.is_dir()]
    timeslides = list(timeslides)
    assert len(timeslides) == n_slides

    for slide in timeslides:
        injection_ts = TimeSlide(slide, field="injection")
        background_ts = TimeSlide(slide, field="background")

        assert len(injection_ts.segments) == len(background_ts.segments) == 1
        segment = background_ts.segments[0]
        h, _ = segment.load("H1")
        l, _ = segment.load("L1")

        expected_size = (1000 - n_slides) * sample_rate
        assert len(h) == len(l) == expected_size

        h_expected = np.arange(expected_size)
        assert (h == h_expected).all()

        l_shift = float(slide.name.split("-L")[-1]) * sample_rate
        l_expected = np.arange(l_shift, l_shift + expected_size)
        assert (l == l_expected).all()

        assert (injection_ts.path / "params.h5").exists()


@patch("bbhnet.parallelize.AsyncExecutor.__enter__", return_value=pool_mock)
@patch("bbhnet.parallelize.AsyncExecutor.__exit__", new=mock_exit)
def test_timeslide_injections_chunked_segments(
    mock1,
    logdir,
    datadir,
    prior,
    spacing,
    jitter,
    buffer_,
    n_slides,
    shifts,
    ifos,
    minimum_frequency,
    highpass,
    sample_rate,
    frame_type,
    channel,
    cosmology,
    state_flag,
):
    chunk_length = 100
    start = 1000000000
    stop = 1000001000

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)

    # expect first segment to remain
    # second to be split into 2 segments
    # third to be split into 6 segments
    segment_list = SegmentList(
        [
            Segment([start + 1, start + 90]),
            Segment([start + 200, start + 400]),
            Segment([start + 400, stop]),
        ]
    )

    segments = DataQualityDict()
    for ifo in ifos:
        segments[f"{ifo}:{state_flag}"] = DataQualityFlag(active=segment_list)

    def fake_read(*args, **kwargs):
        start, stop = kwargs["start"], kwargs["end"]
        t = np.arange(start, stop, 1 / sample_rate)
        x = np.arange(len(t)).astype("float64")
        return TimeSeries(x, times=t)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", new=fake_read)
    mock_datafind = patch("gwdatafind.find_urls", return_value=None)
    mock_segments = patch(
        "gwpy.segments.DataQualityDict.query_dqsegdb", return_value=segments
    )

    with mock_datafind, mock_ts, mock_segments:
        generate_timeslides(
            start=start,
            stop=stop,
            logdir=logdir,
            datadir=datadir,
            prior=prior,
            spacing=spacing,
            jitter=1,
            buffer_=buffer_,
            n_slides=n_slides,
            shifts=shifts,
            ifos=ifos,
            minimum_frequency=minimum_frequency,
            highpass=highpass,
            sample_rate=sample_rate,
            frame_type=frame_type,
            channel=channel,
            cosmology=cosmology,
            state_flag=state_flag,
            chunk_length=chunk_length,
        )

    # get all the timeslide directories
    timeslides = datadir.iterdir()
    timeslides = [slide.name for slide in timeslides if slide.is_dir()]
    timeslides = list(timeslides)
    # create timeslide
    injection_ts = TimeSlide(datadir / "dt-H0.0-L0.0", field="injection")
    background_ts = TimeSlide(datadir / "dt-H0.0-L0.0", field="background")
    assert len(injection_ts.segments) == len(background_ts.segments) == 9

    i = 0
    for slide in datadir.iterdir():
        if not slide.is_dir():
            continue

        i += 1
        for field in ["background", "injection"]:
            ts = TimeSlide(slide, field=field)
            assert len(ts.segments) == 9

        # should be able to find same segment
        # in different time slide
        for segment in injection_ts.segments:
            segment.make_shift(slide.name)

        for segment in background_ts.segments:
            segment.make_shift(slide.name)
    assert i == n_slides


@patch("bbhnet.parallelize.AsyncExecutor.__enter__", return_value=pool_mock)
@patch("bbhnet.parallelize.AsyncExecutor.__exit__", new=mock_exit)
def test_timeslide_injections_with_segments(
    mock1,
    logdir,
    datadir,
    prior,
    spacing,
    jitter,
    buffer_,
    n_slides,
    shifts,
    ifos,
    minimum_frequency,
    highpass,
    sample_rate,
    frame_type,
    channel,
    state_flag,
    cosmology,
):
    start = 1000000000
    stop = 1000001000

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)

    # create same segments for each ifo
    # TODO: test making them shorter thann min_segment_length
    segment_list = SegmentList(
        [
            Segment([start + 200, start + 300]),
            Segment([start + 400, start + 500]),
        ]
    )

    segments = DataQualityDict()
    for ifo in ifos:
        segments[f"{ifo}:{state_flag}"] = DataQualityFlag(active=segment_list)

    def fake_read(*args, **kwargs):
        start, stop = kwargs["start"], kwargs["end"]
        t = np.arange(start, stop, 1 / sample_rate)
        x = np.arange(len(t)).astype("float64")
        return TimeSeries(x, times=t)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", new=fake_read)
    mock_datafind = patch("gwdatafind.find_urls", return_value=None)
    mock_segments = patch(
        "gwpy.segments.DataQualityDict.query_dqsegdb", return_value=segments
    )

    with mock_datafind, mock_ts, mock_segments:
        generate_timeslides(
            start=start,
            stop=stop,
            logdir=logdir,
            datadir=datadir,
            prior=prior,
            spacing=spacing,
            jitter=jitter,
            buffer_=buffer_,
            n_slides=n_slides,
            shifts=shifts,
            ifos=ifos,
            minimum_frequency=minimum_frequency,
            highpass=highpass,
            sample_rate=sample_rate,
            frame_type=frame_type,
            channel=channel,
            cosmology=cosmology,
            state_flag=state_flag,
        )

    # get all the timeslide directories
    timeslides = datadir.iterdir()
    timeslides = [slide.name for slide in timeslides if slide.is_dir()]
    timeslides = list(timeslides)

    # create timeslide
    injection_ts = TimeSlide(datadir / "dt-H0.0-L0.0", field="injection")
    background_ts = TimeSlide(datadir / "dt-H0.0-L0.0", field="background")
    assert len(injection_ts.segments) == len(background_ts.segments) == 2

    i = 0
    for slide in datadir.iterdir():
        if not slide.is_dir():
            continue

        i += 1
        for field in ["background", "injection"]:
            ts = TimeSlide(slide, field=field)
            assert len(ts.segments) == 2

        # should be able to find same segment
        # in different time slide
        for segment in injection_ts.segments:
            segment.make_shift(slide.name)

        for segment in background_ts.segments:
            segment.make_shift(slide.name)
    assert i == n_slides
