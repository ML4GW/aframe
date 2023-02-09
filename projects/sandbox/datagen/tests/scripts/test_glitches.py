from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from datagen.scripts.glitches import generate_glitch_dataset
from gwpy.timeseries import TimeSeries, TimeSeriesDict


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
def trigger_files(ifo):
    fname = Path(__file__).resolve().parent / "triggers" / f"triggers_{ifo}.h5"
    return [str(fname)]


def test_generate_glitch_dataset(
    ifo,
    window,
    sample_rate,
    snr_thresh,
    trigger_files,
    channel,
):
    start = 1263588390
    stop = 1263592390

    glitch_len = 2 * window * sample_rate

    # create mock gwpy timeseries, gwdatafind
    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeriesDict()
    ts[f"{ifo}:{channel}"] = TimeSeries(np.ones(n_samples), times=times)
    mock_ts = patch("gwpy.timeseries.TimeSeriesDict.get", return_value=ts)

    with mock_ts:
        glitches, snrs = generate_glitch_dataset(
            ifo,
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            channel,
            trigger_files,
        )

    assert glitches.shape[-1] == glitch_len
    assert len(glitches) == len(snrs)
    assert all(np.array(snrs) > snr_thresh)
