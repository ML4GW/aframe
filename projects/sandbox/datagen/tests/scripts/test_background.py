from unittest.mock import patch

import gwpy
import h5py
import numpy as np
import pytest
from datagen.scripts import generate_background
from gwpy.segments import DataQualityDict, DataQualityFlag
from gwpy.timeseries import TimeSeries


@pytest.fixture(params=[["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def sample_rate(request):
    return request.param


@patch.object(gwpy.segments.DataQualityDict, "query_dqsegdb")
def test_generate_background(
    mock_query,
    datadir,
    logdir,
    ifos,
    sample_rate,
):
    start = 1234567890
    stop = 1234577890

    # construct mock DataQualityDict
    # object that just contains the full
    # start to stop as its only segment
    state_flag = "DCS-ANALYSIS_READY_C01:1"
    segments = DataQualityFlag(active=[[start, stop]])
    segment_list = DataQualityDict()
    for ifo in ifos:
        segment_list[f"{ifo}:{state_flag}"] = segments

    mock_query.return_value = segment_list

    minimum_length = 10
    channel = "DCS-CALIB_STRAIN_CLEAN_C01"
    frame_type = "HOFT_C01"

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)
    mock_datafind = patch(
        "datagen.scripts.background.find_urls", return_value=None
    )
    with mock_ts, mock_datafind:
        generate_background(
            start,
            stop,
            ifos,
            sample_rate,
            channel,
            frame_type,
            state_flag,
            minimum_length,
            datadir,
            logdir,
        )

    for ifo in ifos:
        background_path = datadir / f"{ifo}_background.h5"
        with h5py.File(background_path) as f:
            assert (f["hoft"] == ts.value).all()
            assert f["t0"][()] == start
