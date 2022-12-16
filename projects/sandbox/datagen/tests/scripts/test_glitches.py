from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from datagen.scripts.glitches import generate_glitch_dataset, veto
from gwpy.segments import Segment, SegmentList
from gwpy.timeseries import TimeSeries


@pytest.fixture(params=[2, 4])
def window(request):
    return request.param


@pytest.fixture(params=["HOFT_C01"])
def frame_type(request):
    return request.param


@pytest.fixture(params=["DCS-CALIB_STRAIN_CLEAN_C01"])
def channel(request):
    return request.param


@pytest.fixture(params=[11, 15])
def snr_thresh(request):
    return request.param


@pytest.fixture(params=["H1", "L1"])
def ifo(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture
def trig_file(ifo):
    fname = Path(__file__).resolve().parent / "triggers" / f"triggers_{ifo}.h5"
    return str(fname)


def test_generate_glitch_dataset(
    ifo,
    window,
    sample_rate,
    snr_thresh,
    trig_file,
    channel,
    frame_type,
):
    start = 1263588390
    stop = 1263592390

    glitch_len = 2 * window * sample_rate

    # create mock gwpy timeseries, gwdatafind
    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)
    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)
    mock_datafind = patch("gwdatafind.find_urls", return_value=None)

    with mock_ts, mock_datafind:
        glitches, snrs = generate_glitch_dataset(
            ifo,
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            channel,
            frame_type,
            trig_file,
        )

    assert glitches.shape[-1] == glitch_len
    assert len(glitches) == len(snrs)
    assert all(snrs > snr_thresh)


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
