import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from bbhnet.io.h5 import write_timeseries


@pytest.fixture
def data_dir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture
def out_dir():
    tmpdir = Path(__file__).resolve().parent / "out"
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


@pytest.fixture(params=[1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture
def arange_background(data_dir, sample_rate):
    def func(data_dir, sample_rate, length, ifo):
        times = np.arange(1, length * sample_rate, 1)
        background = np.random.normal(size=len(times))

        path = write_timeseries(data_dir, "background", times, hoft=background)
        return path

    return func


@pytest.fixture
def arange_glitches(data_dir, sample_rate):
    def func(data_dir, sample_rate, duration, num_glitches):
        glitches = np.arange(sample_rate * duration * num_glitches).reshape(
            num_glitches, sample_rate * duration
        )

        path = data_dir / "glitches.h5"
        with h5py.File(path, "w") as f:
            f["H1_glitches"] = glitches
            f["L1_glitches"] = -1 * glitches

        return path

    return func


@pytest.fixture
def arange_waveforms(data_dir, sample_rate):
    def func(data_dir, sample_rate, duration, num_waveforms):
        n_pols = 2
        waveforms = np.arange(
            n_pols * sample_rate * duration * num_waveforms
        ).reshape(num_waveforms, n_pols, sample_rate * duration)

        path = data_dir / "waveforms.h5"
        with h5py.File(path, "w") as f:
            f["signals"] = waveforms
            f["dec"] = np.zeros(len(waveforms))
            f["ra"] = np.zeros(len(waveforms))
            f["psi"] = np.zeros(len(waveforms))
        return path

    return func
