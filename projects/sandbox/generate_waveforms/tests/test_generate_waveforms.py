import logging
import shutil
from pathlib import Path

import h5py
import pytest
from generate_waveforms import main

from bbhnet.injection import end_o3_ratesandpops, nonspin_bbh


@pytest.fixture
def data_dir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture
def log_dir():
    tmpdir = Path(__file__).resolve().parent / "log"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture(params=[0, 10, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[50])
def reference_frequency(request):
    return request.param


@pytest.fixture(params=[20, 40])
def minimum_frequency(request):
    return request.param


@pytest.fixture(params=[8])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[512, 4096, 16384])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[nonspin_bbh, end_o3_ratesandpops])
def prior(request):
    return request.param


def test_check_file_contents(
    data_dir,
    log_dir,
    n_samples,
    waveform_duration,
    sample_rate,
    prior,
    minimum_frequency,
    reference_frequency,
):

    signal_length = waveform_duration * sample_rate

    signal_file = main(
        prior,
        n_samples,
        log_dir,
        data_dir,
        reference_frequency,
        minimum_frequency,
        sample_rate,
        waveform_duration,
    )

    with h5py.File(signal_file, "r") as f:
        for key in f.keys():
            if key == "signals":
                act_shape = f[key].shape
                exp_shape = (n_samples, 2, signal_length)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for signals, found {act_shape}"
            else:
                act_shape = f[key].shape
                exp_shape = (n_samples,)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for {key}, found {act_shape}"
