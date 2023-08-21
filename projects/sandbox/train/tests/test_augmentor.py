from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from train.augmentor import AframeBatchAugmentor

from aframe.architectures.preprocessor import PsdEstimator


@pytest.fixture
def sample_rate():
    return 128


@pytest.fixture
def kernel_length():
    return 1


@pytest.fixture
def background_length():
    return 10


@pytest.fixture
def fduration():
    return 1


# total size of kernels that get sampled
@pytest.fixture
def size(sample_rate, kernel_length, background_length, fduration):
    return int(sample_rate * (background_length + kernel_length + fduration))


def sample(obj, N, kernel_size, _):
    return torch.ones((N, 2, kernel_size))


# dummy psd estimator and whitener
@pytest.fixture
def psd_estimator(kernel_length, fduration, sample_rate):
    return PsdEstimator(kernel_length + fduration, sample_rate, 2)


@pytest.fixture
def whitener():
    def skip(X, psds):
        return X * 2

    return skip


rand_value = 0.1 + 0.5 * (torch.arange(32) % 2)


@patch("train.augmentor.AframeBatchAugmentor.sample_responses", new=sample)
@patch("torch.rand", return_value=rand_value)
def test_bbhnet_batch_augmentor(
    rand_mock, psd_estimator, whitener, size, sample_rate
):
    """
    glitch_sampler = Mock(
        return_value=(torch.zeros((32, 2, 128 * 1)), torch.zeros((32, 1)))
    )
    glitch_sampler.prob = 0.0
    """

    augmentor = AframeBatchAugmentor(
        ifos=["H1", "L1"],
        sample_rate=sample_rate,
        signal_prob=0.5,
        whitener=whitener,
        dec=MagicMock(),
        psi=MagicMock(),
        phi=MagicMock(),
        trigger_distance=-0.5,
        plus=np.zeros((1, 128 * 2)),
        cross=np.zeros((1, 128 * 2)),
        psd_estimator=psd_estimator,
    )

    X = torch.zeros((32, 2, size))
    y = torch.zeros((32, 1))

    X, y = augmentor(X, y)
    assert (X[::2] == 2).all().item()
    assert (X[1::2] == 0).all().item()
    assert (y[::2] == 1).all().item()
    assert (y[1::2] == 0).all().item()

    for _, tensor in augmentor.polarizations.items():
        assert len(tensor) == augmentor.num_waveforms

    # 2 buffers for the ifo geometry
    # 2 buffers belong to the whitener
    assert len(list(augmentor.buffers())) == 4

    polarizations = {
        "plus": torch.randn(100, 4096),
        "cross": torch.randn(99, 4096),
    }
    with pytest.raises(ValueError) as exc:
        augmentor = AframeBatchAugmentor(
            ifos=["H1", "L1"],
            sample_rate=sample_rate,
            signal_prob=0.9,
            psd_estimator=psd_estimator,
            whitener=whitener,
            dec=lambda N: torch.randn(N, 3, 4096),
            psi=lambda N: torch.randn(N, 15),
            phi=lambda N: torch.randn(N, 15),
            trigger_distance=0.1,
            **polarizations,
        )

    assert str(exc.value).startswith("Polarization ")


@patch("train.augmentor.AframeBatchAugmentor.sample_responses", new=sample)
@pytest.mark.parametrize("swap_frac", [0, 0.1])
@pytest.mark.parametrize("mute_frac", [0, 0.1])
def test_bbhnet_batch_augmentor_with_swapping_and_muting(
    size,
    swap_frac,
    mute_frac,
    psd_estimator,
    whitener,
    kernel_length,
    fduration,
    sample_rate,
):

    X = torch.zeros((32, 2, size))
    y = torch.zeros((32, 1))

    if swap_frac == 0.5:
        with pytest.raises(ValueError) as exc:
            augmentor = AframeBatchAugmentor(
                ifos=["H1", "L1"],
                sample_rate=sample_rate,
                signal_prob=0.9,
                # glitch_sampler=glitch_sampler,
                dec=lambda N: torch.zeros((N,)),
                psi=lambda N: torch.zeros((N,)),
                phi=lambda N: torch.zeros((N,)),
                trigger_distance=-0.5,
                swap_frac=swap_frac,
                mute_frac=mute_frac,
                plus=np.zeros((1, sample_rate * 8)),
                cross=np.zeros((1, sample_rate * 8)),
                psd_estimator=psd_estimator,
                whitener=whitener,
            )
        assert str(exc.value).startswith("Probability must be")

    X = torch.zeros((32, 2, size))
    y = torch.zeros((32, 1))
    augmentor = AframeBatchAugmentor(
        sample_rate=sample_rate,
        ifos=["H1", "L1"],
        signal_prob=0.5,
        trigger_distance=-0.25,
        dec=lambda N: torch.zeros((N,)),
        psi=lambda N: torch.zeros((N,)),
        phi=lambda N: torch.zeros((N,)),
        swap_frac=swap_frac,
        mute_frac=mute_frac,
        plus=np.zeros((100, 128 * 2)),
        cross=np.zeros((100, 128 * 2)),
        psd_estimator=psd_estimator,
        whitener=whitener,
    )

    if swap_frac + mute_frac == 0:
        assert augmentor.signal_prob == 0.5
    elif swap_frac + mute_frac == 0.1:
        assert augmentor.signal_prob == (0.5 / (1 - 0.1))
    else:
        assert augmentor.signal_prob == (0.5 / (1 - (0.2 - 0.01)))

    value = 0.99 * augmentor.signal_prob
    mute_indices = torch.arange(4, 8) if mute_frac == 0.1 else []
    swap_indices = torch.arange(4) if swap_frac == 0.1 else []

    waveform_shape = torch.randn(
        X.shape[0], 2, (kernel_length + fduration) * sample_rate
    )
    mock_channel_muter = MagicMock(return_value=(waveform_shape, mute_indices))
    mock_channel_swapper = MagicMock(
        return_value=(waveform_shape, swap_indices)
    )

    augmentor.muter.forward = mock_channel_muter
    augmentor.swapper.forward = mock_channel_swapper

    with patch("torch.rand", return_value=value):
        X, y = augmentor(X, y)

        if swap_frac + mute_frac == 0:
            assert (y > 0).all().item()
        elif swap_frac == 0 and mute_frac == 0.1:
            assert (y[4:8] == 0).all().item()
            assert (y[:4] == 1).all().item()
            assert (y[8:] == 1).all().item()
        elif swap_frac == 0.1 and mute_frac == 0:
            assert (y[:4] == 0).all().item()
            assert (y[4:] == 1).all().item()
        else:
            assert (y[:8] == 0).all().item()
            assert (y[8:] == 1).all().item()


def test_sample_responses(
    psd_estimator, whitener, background_length, sample_rate
):
    # Test that sample_responses returns the expected output shape
    ifos = ["H1", "L1"]

    signal_prob = 0.5

    polarizations = {
        "plus": np.random.randn(100, 4096 * 2),
        "cross": np.random.randn(100, 4096 * 2),
    }
    augmentor = AframeBatchAugmentor(
        ifos,
        sample_rate,
        signal_prob,
        dec=lambda N: torch.randn(N),
        psi=lambda N: torch.randn(N),
        phi=lambda N: torch.randn(N),
        trigger_distance=-0.5,
        psd_estimator=psd_estimator,
        whitener=whitener,
        **polarizations,
    )

    N = 10
    kernel_size = 4096
    psds = torch.ones((N, 2, background_length * sample_rate))
    kernels = augmentor.sample_responses(N, kernel_size, psds)
    assert kernels.shape == (N, 2, kernel_size)
