import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def data_length():
    return 128


@pytest.fixture(params=[True, False])
def deterministic(request):
    return request.param


@pytest.fixture
def offset():
    # TODO: explore more values
    return 0


@pytest.fixture(params=[None, -0.2, 0.5, 1])
def frac(request):
    return request.param


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    return request.param


@pytest.fixture(scope="function")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4])
def glitch_length(request):
    return request.param


@pytest.fixture
def write_timeseries(data_dir):
    def func(fname, **kwargs):
        with h5py.File(data_dir / fname, "w") as f:
            for key, value in kwargs.items():
                f[key] = value
        return data_dir / fname

    return func


@pytest.fixture
def arange_glitches(glitch_length, sample_rate, write_timeseries, data_dir):
    num_glitches = 100
    glitches = np.arange(num_glitches * glitch_length * sample_rate).reshape(
        num_glitches, -1
    )

    data = {
        "H1_glitches": glitches,
        "L1_glitches": -glitches,
    }
    write_timeseries("arange_glitches.h5", **data)
    return data_dir / "arange_glitches.h5"


@pytest.fixture(params=[["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def ifos(request):
    return request.param


@pytest.fixture
def sine_waveforms(
    glitch_length, sample_rate, write_timeseries, data_dir, ifos
):
    x = np.linspace(0, 4 * np.pi, glitch_length * sample_rate)
    waveforms = np.stack([np.sin(i * 2 * np.pi * x) for i in range(1, 101)])

    # need two polarizations
    waveforms = waveforms[:, None]
    waveforms = [waveforms, waveforms * 0.5]
    waveforms = np.concatenate(waveforms, axis=1)
    write_timeseries(
        "sine_waveforms.h5",
        signals=waveforms,
        ra=np.zeros((10,)),
        dec=np.zeros((10,)),
        psi=np.zeros((10,)),
        geocent_time=np.ones((10,)) * 1234567890,
    )
    return data_dir / "sine_waveforms.h5"
