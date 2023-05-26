from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from train.augmentor import AframeBatchAugmentor


@pytest.fixture
def sample_rate():
    return 512


def sample(obj, N, kernel_size):
    return torch.ones((N, 2, kernel_size))


rand_value = 0.1 + 0.5 * (torch.arange(32) % 2)


@patch("train.augmentor.AframeBatchAugmentor.sample_responses", new=sample)
@patch("torch.rand", return_value=rand_value)
def test_bbhnet_batch_augmentor(rand_mock):

    glitch_sampler = Mock(
        return_value=(torch.zeros((32, 2, 128 * 1)), torch.zeros((32, 1)))
    )
    glitch_sampler.prob = 0.0

    augmentor = AframeBatchAugmentor(
        ifos=["H1", "L1"],
        sample_rate=2048,
        signal_prob=0.5,
        glitch_sampler=glitch_sampler,
        dec=MagicMock(),
        psi=MagicMock(),
        phi=MagicMock(),
        trigger_distance=-0.5,
        plus=np.zeros((1, 128 * 2)),
        cross=np.zeros((1, 128 * 2)),
    )

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))

    X, y = augmentor(X, y)
    assert (X[::2] == 1).all().item()
    assert (X[1::2] == 0).all().item()
    assert (y[::2] == 1).all().item()
    assert (y[1::2] == 0).all().item()

    for _, tensor in augmentor.polarizations.items():
        assert len(tensor) == augmentor.num_waveforms

    assert len(list(augmentor.buffers())) == 2

    polarizations = {
        "plus": torch.randn(100, 4096),
        "cross": torch.randn(99, 4096),
    }
    with pytest.raises(ValueError) as exc:
        augmentor = AframeBatchAugmentor(
            ifos=["H1", "L1"],
            sample_rate=2048,
            signal_prob=0.9,
            glitch_sampler=glitch_sampler,
            dec=lambda N: torch.randn(N, 3, 4096),
            psi=lambda N: torch.randn(N, 15),
            phi=lambda N: torch.randn(N, 15),
            trigger_distance=0.1,
            **polarizations,
        )

    assert str(exc.value).startswith("Polarization ")


@pytest.mark.parametrize("downweight", [0, 0.5, 1])
def test_bbhnet_batch_augmentor_with_downweight(downweight):
    glitch_sampler = Mock(
        return_value=(torch.zeros((32, 2, 128 * 1)), torch.zeros((32, 1)))
    )
    glitch_sampler.prob = 0.25
    if downweight == 0.5:
        with pytest.raises(ValueError) as exc:
            augmentor = AframeBatchAugmentor(
                ifos=["H1", "L1"],
                sample_rate=2048,
                signal_prob=0.9,
                glitch_sampler=glitch_sampler,
                dec=lambda N: torch.zeros((N,)),
                psi=lambda N: torch.zeros((N,)),
                phi=lambda N: torch.zeros((N,)),
                trigger_distance=-0.5,
                downweight=downweight,
                plus=np.zeros((1, 128 * 2)),
                cross=np.zeros((1, 128 * 2)),
            )
        assert str(exc.value).startswith("Probability must be")

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))
    y[:8] = -2
    y[8:16] = -4
    y[16:24] = -6

    glitch_sampler = Mock(return_value=(X, y))
    glitch_sampler.prob = 0.25
    augmentor = AframeBatchAugmentor(
        sample_rate=128,
        ifos=["H1", "L1"],
        signal_prob=0.5,
        glitch_sampler=glitch_sampler,
        trigger_distance=-0.25,
        dec=lambda N: torch.zeros((N,)),
        psi=lambda N: torch.zeros((N,)),
        phi=lambda N: torch.zeros((N,)),
        downweight=downweight,
        plus=np.zeros((100, 128 * 2)),
        cross=np.zeros((100, 128 * 2)),
    )

    if downweight == 1:
        assert augmentor.signal_prob == 0.5
    elif downweight == 0:
        assert augmentor.signal_prob == (0.5 / 0.75**2)
    else:
        assert augmentor.signal_prob > 0.5

    value = 0.99 * augmentor.signal_prob
    if (downweight != 0) and (downweight) != 1:
        value = value * downweight
    with patch("torch.rand", return_value=value):
        X, y = augmentor(X, y)
        if downweight == 1:
            assert (y > 0).all().item()
        elif downweight == 0:
            assert (y[:24] < 0).all().item()
            assert (y[24:] == 1).all().item()
        else:
            assert (y[:16] > 0).all().item()
            assert (y[16:24] < 0).all().item()
            assert (y[24:] > 0).all().item()


@patch("train.augmentor.AframeBatchAugmentor.sample_responses", new=sample)
@pytest.mark.parametrize("swap_frac", [0, 0.1])
@pytest.mark.parametrize("mute_frac", [0, 0.1])
def test_bbhnet_batch_augmentor_with_swapping_and_muting(swap_frac, mute_frac):

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))
    glitch_sampler = Mock(return_value=(X, y))
    glitch_sampler.prob = 0.25

    if swap_frac == 0.5:
        with pytest.raises(ValueError) as exc:
            augmentor = AframeBatchAugmentor(
                ifos=["H1", "L1"],
                sample_rate=2048,
                signal_prob=0.9,
                glitch_sampler=glitch_sampler,
                dec=lambda N: torch.zeros((N,)),
                psi=lambda N: torch.zeros((N,)),
                phi=lambda N: torch.zeros((N,)),
                trigger_distance=-0.5,
                swap_frac=swap_frac,
                mute_frac=mute_frac,
                plus=np.zeros((1, 128 * 2)),
                cross=np.zeros((1, 128 * 2)),
            )
        assert str(exc.value).startswith("Probability must be")

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))
    glitch_sampler = Mock(return_value=(X, y))
    glitch_sampler.prob = 0.0

    augmentor = AframeBatchAugmentor(
        sample_rate=128,
        ifos=["H1", "L1"],
        signal_prob=0.5,
        glitch_sampler=glitch_sampler,
        trigger_distance=-0.25,
        dec=lambda N: torch.zeros((N,)),
        psi=lambda N: torch.zeros((N,)),
        phi=lambda N: torch.zeros((N,)),
        swap_frac=swap_frac,
        mute_frac=mute_frac,
        plus=np.zeros((100, 128 * 2)),
        cross=np.zeros((100, 128 * 2)),
    )

    if swap_frac + mute_frac == 0:
        assert augmentor.signal_prob == 0.5
    elif swap_frac + mute_frac == 0.1:
        assert augmentor.signal_prob == (0.5 / (1 - 0.1))
    else:
        assert augmentor.signal_prob == (0.5 / (1 - (0.2 - 0.01)))

    #
    value = 0.99 * augmentor.signal_prob
    mute_indices = torch.arange(4, 8) if mute_frac == 0.1 else []
    swap_indices = torch.arange(4) if swap_frac == 0.1 else []

    mock_channel_muter = MagicMock(return_value=(X, mute_indices))
    mock_channel_swapper = MagicMock(return_value=(X, swap_indices))

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


def test_sample_responses():
    # Test that sample_responses returns the expected output shape
    ifos = ["H1", "L1"]
    sample_rate = 2048
    signal_prob = 0.5
    glitch_sampler = MagicMock(return_value=torch.randn(10, 3, 4096))
    glitch_sampler.prob = 0.25

    polarizations = {
        "plus": np.random.randn(100, 4096 * 2),
        "cross": np.random.randn(100, 4096 * 2),
    }
    augmentor = AframeBatchAugmentor(
        ifos,
        sample_rate,
        signal_prob,
        glitch_sampler,
        dec=lambda N: torch.randn(N),
        psi=lambda N: torch.randn(N),
        phi=lambda N: torch.randn(N),
        trigger_distance=-0.5,
        **polarizations,
    )

    N = 10
    kernel_size = 4096
    kernels = augmentor.sample_responses(N, kernel_size)
    assert kernels.shape == (N, 2, kernel_size)
