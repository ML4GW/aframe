#!/usr/bin/env python
# coding: utf-8
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from generate_glitches import generate_glitch_dataset
from gwpy.timeseries import TimeSeries

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


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


@pytest.fixture()
def omicron_dir(request):
    return "/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/"


@pytest.fixture()
def trig_file(ifo):
    return str(TEST_DIR / "triggers" / f"triggers_{ifo}.h5")


def test_glitch_data_shape_and_glitch_snrs(
    data_dir,
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
