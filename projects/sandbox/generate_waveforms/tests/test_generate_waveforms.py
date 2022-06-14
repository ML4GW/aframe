import logging
import shutil
from pathlib import Path

import h5py
import pytest
from generate_waveforms import main


@pytest.fixture  # (scope="session")
def data_dir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture(params=[0, 10, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[1, 8, 60])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[512, 4096, 16384])
def sample_rate(request):
    return request.param


@pytest.fixture(
    params=["prior_files/nonspin_BBH.prior", "prior_files/precess_tides.prior"]
)
def prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


def test_check_file_contents(
    data_dir, n_samples, waveform_duration, sample_rate, prior_file
):

    signal_length = waveform_duration * sample_rate

    signal_file = main(
        prior_file,
        n_samples,
        data_dir,
        waveform_duration=waveform_duration,
        sample_rate=sample_rate,
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
