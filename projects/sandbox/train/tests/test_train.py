import logging
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from train.train import main as train

from ml4gw.gw import compute_network_snr


@pytest.fixture(params=[1024])
def sample_rate(request):
    return request.param


@pytest.fixture
def num_non_background():
    return 500


@pytest.fixture
def non_background_length():
    return 4


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture
def duration():
    return 10000


@pytest.fixture
def background(duration, sample_rate):
    times = np.arange(1, duration * sample_rate, 1)
    background = np.random.normal(size=len(times))
    return background, times


@pytest.fixture
def glitches(sample_rate, num_non_background, non_background_length):
    size = sample_rate * non_background_length
    glitches = np.arange(size * num_non_background)
    glitches = glitches.reshape(num_non_background, size)
    return glitches


@pytest.fixture
def glitch_times(num_non_background, duration):
    return np.linspace(0, duration, num_non_background)


@pytest.fixture
def waveforms(sample_rate, num_non_background, non_background_length):
    size = sample_rate * non_background_length
    waveforms = np.arange(2 * size * num_non_background)
    waveforms = waveforms.reshape(num_non_background, 2, size)
    return waveforms


@pytest.fixture
def h5py_mock(background, glitches, glitch_times, waveforms, ifos):
    def mock(fname, _):
        if "background" in fname:
            hoft, times = background
            value = {}
            for ifo in ifos:
                dataset = MagicMock()
                dataset.__getitem__.side_effect = hoft.__getitem__
                dataset.attrs = {"x0": times[0]}
                value[ifo] = dataset
        elif "glitches" in fname:
            value = {ifo: {} for ifo in ifos}
            for i, ifo in enumerate(ifos):
                sign = (-1) ** i
                value[ifo]["glitches"] = sign * glitches
                value[ifo]["times"] = glitch_times
        elif "signals" in fname:
            zeros = np.zeros((len(waveforms),))
            value = {i: zeros for i in ["dec", "ra", "psi"]}
            value["signals"] = waveforms
        else:
            raise ValueError(fname)

        obj = Mock()
        obj.__enter__ = lambda obj: value
        obj.__exit__ = Mock()
        return obj

    with patch("h5py.File", new=mock):
        yield mock


@pytest.fixture(params=[0.25, 0.67])
def glitch_prob(request):
    return request.param


@pytest.fixture(params=[0.2, 0.6])
def valid_frac(request):
    return request.param


@pytest.fixture
def outdir(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    yield outdir
    logging.shutdown()


def test_train(
    outdir,
    duration,
    background,
    h5py_mock,
    sample_rate,
    glitch_prob,
    valid_frac,
    num_non_background,
    non_background_length,
):
    num_waveforms = num_non_background
    num_ifos = 2
    kernel_length = 2
    fduration = 1

    background_dir = MagicMock()
    background_dir.iterdir = lambda: iter(["background.h5"])
    train_dataset, validator, preprocessor = train(
        background_dir,
        "glitches.h5",
        "signals.h5",
        outdir,
        outdir,
        glitch_prob=glitch_prob,
        waveform_prob=0.5,
        glitch_downweight=0.8,
        snr_thresh=6,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=512,
        max_min_snr=12,
        min_min_snr=4,
        max_snr=100,
        snr_alpha=3,
        snr_decay_steps=5000,
        mean_snr=15,
        std_snr=15,
        min_snr=1,
        highpass=32,
        batches_per_epoch=200,
        # preproc args
        fduration=fduration,
        trigger_distance=-0.5,
        # validation args
        valid_frac=valid_frac,
        valid_stride=1,
    )

    background, _ = background
    background = np.stack([background] * 2)

    # Check that the training background is what it should be
    train_frac = 1 - valid_frac
    train_length = int(duration * train_frac * sample_rate - 1)
    train_background = background[:, :train_length]
    np.testing.assert_allclose(train_dataset.X.numpy(), train_background)

    # Validator
    # check that we have the expected amount of validation data
    assert validator.duration == (duration * valid_frac)

    # check that we have the expected number of waveforms
    num_valid_waveforms = int(valid_frac * num_waveforms)
    waveform_size = int(non_background_length * sample_rate)
    assert validator.waveforms.shape == (
        num_valid_waveforms,
        num_ifos,
        waveform_size,
    )

    # check that our waveforms all have the correct snrs
    psd = train_dataset.preprocessor.rescaler.background
    snrs = compute_network_snr(
        validator.waveforms, psd, sample_rate, highpass=32
    )
    assert (snrs > 6).all().item

    # Preprocessor
    # Check that the whitening filter is the correct shape
    assert (
        preprocessor.whitener.time_domain_filter.numpy().shape
        == np.array((num_ifos, 1, fduration * sample_rate - 1))
    ).all()
