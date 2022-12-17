import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
from train.train import main as train


@pytest.fixture(params=[1024])
def sample_rate(request):
    return request.param


@pytest.fixture
def num_non_background():
    return 500


@pytest.fixture
def non_background_length():
    return 4


@pytest.fixture
def background(sample_rate):
    length = 10000
    times = np.arange(1, length * sample_rate, 1)
    background = np.random.normal(size=len(times))
    return background, times


@pytest.fixture
def glitches(sample_rate, num_non_background, non_background_length):
    size = sample_rate * non_background_length
    glitches = np.arange(size * num_non_background)
    glitches = glitches.reshape(num_non_background, size)
    return glitches


@pytest.fixture
def waveforms(sample_rate, num_non_background, non_background_length):
    size = sample_rate * non_background_length
    waveforms = np.arange(2 * size * num_non_background)
    waveforms = waveforms.reshape(num_non_background, 2, size)
    return waveforms


@pytest.fixture
def h5py_mock(background, glitches, waveforms):
    def mock(fname, _):
        if "background" in fname:
            hoft, times = background
            value = {"hoft": hoft, "t": times}
        elif "glitches" in fname:
            value = {"H1_glitches": glitches, "L1_glitches": -glitches}
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
    background,
    h5py_mock,
    sample_rate,
    glitch_prob,
    valid_frac,
    num_non_background,
    non_background_length,
    waveforms,
):
    num_waveforms = num_glitches = num_non_background
    duration = 10000
    num_ifos = 2
    kernel_length = 2
    fduration = 1

    train_dataset, validator, preprocessor = train(
        "H1_background.h5",
        "L1_background.h5",
        "glitches.h5",
        "signals.h5",
        outdir,
        outdir,
        glitch_prob=glitch_prob,
        waveform_prob=0.5,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=512,
        mean_snr=10,
        std_snr=4,
        min_snr=2,
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
    # Background DataLoader
    # Check that the background loader has the right shape and data
    num_kernels = (duration * valid_frac) // (kernel_length - fduration) - 1
    num_kernels = int(num_kernels)
    kernel_size = kernel_length * sample_rate
    background_loader = validator.background_loader.dataset[:][0].numpy()

    # Check shape
    assert (
        background_loader.shape
        == np.array((num_kernels, num_ifos, kernel_size))
    ).all()
    # Check that each kernel has mean 0. atol is ~5 sigma
    np.testing.assert_allclose(
        np.mean(background_loader, axis=2),
        np.zeros((num_kernels, num_ifos)),
        atol=1.1e-01,
    )

    # Glitch DataLoader
    # This is set up to avoid any off by 1 errors that come from rounding
    total_valid_glitches = 2 * num_glitches - int(
        train_frac * 2 * num_glitches
    )
    num_coinc = int(
        glitch_prob**2 * total_valid_glitches / (1 + glitch_prob**2)
    )
    num_valid_glitches = total_valid_glitches - num_coinc
    num_non_coinc = num_valid_glitches - num_coinc
    h1_glitches = num_non_coinc // 2
    glitch_loader = validator.glitch_loader.dataset[:][0].numpy()

    # Check shape
    assert (
        glitch_loader.shape
        == np.array((num_valid_glitches, num_ifos, kernel_size))
    ).all()

    # Check that the kernels are aranges (glitches) where we expect them to be
    expected_h1_glitches = np.array(
        [
            np.arange(start, start + kernel_size)
            for start in glitch_loader[:h1_glitches, 0, 0]
        ]
    )
    expected_l1_glitches = -expected_h1_glitches

    expected_h1_coinc = np.array(
        [
            np.arange(start, start + kernel_size)
            for start in glitch_loader[num_non_coinc:, 0, 0]
        ]
    )
    expected_l1_coinc = -expected_h1_coinc
    expected_coinc_glitches = np.stack(
        (expected_h1_coinc, expected_l1_coinc), axis=1
    )

    assert (glitch_loader[:h1_glitches, 0, :] == expected_h1_glitches).all()
    assert (
        glitch_loader[h1_glitches:num_non_coinc, 1, :] == expected_l1_glitches
    ).all()
    assert (
        glitch_loader[num_non_coinc:, :, :] == expected_coinc_glitches
    ).all()

    # Check that the kernels are background where we expect them to be
    np.testing.assert_allclose(
        np.mean(glitch_loader[:h1_glitches, 1, :], axis=1),
        np.zeros(h1_glitches),
        atol=1.1e-01,
    )
    np.testing.assert_allclose(
        np.mean(glitch_loader[h1_glitches:num_non_coinc, 0, :], axis=1),
        np.zeros(h1_glitches),
        atol=1.1e-01,
    )
    print(num_kernels, h1_glitches)

    # Signal DataLoader
    # Check shape
    signal_loader = validator.signal_loader.dataset[:][0].numpy()
    num_valid_waveforms = int(valid_frac * num_waveforms)
    assert (
        signal_loader.shape
        == np.array((num_valid_waveforms, num_ifos, kernel_size))
    ).all()

    # Can't check signal content directly without knowing how the waveforms
    # will be projected onto the ifos. But the polarizations are linear, so
    # the ifo responses should also be linear, at least over a time frame of
    # a couple seconds.
    actual_signals = signal_loader - background_loader[: len(signal_loader)]
    slopes = (actual_signals[:, :, -1] - actual_signals[:, :, 0]) / kernel_size
    mean_slopes = np.mean(slopes, axis=0)
    intercepts = actual_signals[:, :, 0]

    expected_h1_signals = np.array(
        [b + mean_slopes[0] * np.arange(kernel_size) for b in intercepts[:, 0]]
    )
    expected_l1_signals = np.array(
        [b + mean_slopes[1] * np.arange(kernel_size) for b in intercepts[:, 1]]
    )
    expected_signals = np.stack(
        (expected_h1_signals, expected_l1_signals), axis=1
    )

    # There's no precise reaoson for this tolerance
    np.testing.assert_allclose(actual_signals, expected_signals, rtol=1e-6)

    # Preprocessor
    # Check that the whitening filter is the correct shape
    assert (
        preprocessor.whitener.time_domain_filter.numpy().shape
        == np.array((num_ifos, 1, fduration * sample_rate - 1))
    ).all()
