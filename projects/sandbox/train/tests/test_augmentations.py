import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from train.augmentations import (
    ChannelSwapper,
    SignalInverter,
    SignalReverser,
    SnrSampler,
)


@pytest.fixture
def sample_rate():
    return 512


@pytest.fixture(params=[0, 0.25, 1])
def prob(request):
    return request.param


@pytest.fixture(params=[0.0, 0.25, 0.5, 1])
def flip_prob(request):
    return request.param


@pytest.fixture
def rvs():
    return torch.Tensor([[0.0, 0.49], [0.51, 0.1], [0.9, 0.2], [0.1, 0.3]])


@pytest.fixture
def true_idx(flip_prob):
    if flip_prob in (0, 1):
        idx = [k for i in range(4) for k in [[i, j] for j in range(2)]]
        if flip_prob:
            neg_idx = idx
            pos_idx = []
        else:
            pos_idx = idx
            neg_idx = []
    else:
        if flip_prob == 0.5:
            neg_idx = [
                [0, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [3, 0],
                [3, 1],
            ]
            pos_idx = [
                [1, 0],
                [2, 0],
            ]
        else:
            neg_idx = [[0, 0], [1, 1], [2, 1], [3, 0]]
            pos_idx = [
                [0, 1],
                [1, 0],
                [2, 0],
                [3, 1],
            ]

    return neg_idx, pos_idx


def validate_augmenters(X, idx, true, false, prob):
    neg_idx, pos_idx = idx
    if neg_idx:
        neg0, neg1 = zip(*neg_idx)
        assert (X[neg0, neg1] == true).all()
    elif prob != 0:
        raise ValueError("Missing negative indices")

    if pos_idx:
        pos0, pos1 = zip(*pos_idx)
        assert (X[pos0, pos1] == false).all()
    elif prob != 1:
        raise ValueError("Missing positive indices")


def test_signal_inverter(flip_prob, rvs, true_idx):
    tform = SignalInverter(flip_prob)
    X = torch.ones((4, 2, 8))
    with patch("torch.rand", return_value=rvs):
        X = tform(X)
    X = X.cpu().numpy()
    validate_augmenters(X, true_idx, -1, 1, flip_prob)


def test_signal_reverser(flip_prob, rvs, true_idx):
    tform = SignalReverser(flip_prob)
    x = torch.arange(8)
    X = torch.stack([x] * 2)
    X = torch.stack([X] * 4)
    with patch("torch.rand", return_value=rvs):
        X = tform(X)
    X = X.cpu().numpy()

    x = x.cpu().numpy()
    validate_augmenters(X, true_idx, x[::-1], x, flip_prob)


@pytest.fixture(params=[0.5])
def frac():
    return 0.5


def test_channel_swapper(frac):
    tform = ChannelSwapper(frac=frac)
    n_batch = 128
    X = (
        torch.arange(n_batch)
        .repeat(n_batch, 1)
        .transpose(1, 0)
        .reshape(n_batch, 2, -1)
    )
    num = int(frac * n_batch)
    num = num if not num % 2 else num - 1
    channels = torch.ones(num // 2, dtype=torch.long)
    copy = torch.clone(X)
    with patch("torch.randint", return_value=channels):
        X, indices = tform(X)

    target_indices = torch.roll(indices, shifts=num // 2, dims=0)
    X = X.cpu().numpy()
    copy = copy.cpu().numpy()
    indices = indices.cpu().numpy()
    channels = channels.repeat(2)

    assert all(indices == np.arange(num))
    assert (X[indices, channels] == copy[target_indices, channels]).all()

    tform = ChannelSwapper(frac=0)
    X = torch.arange(6).repeat(6, 1).transpose(1, 0).reshape(-1, 2, 6)
    copy = torch.clone(X)
    X, indices = tform(X)
    assert not indices
    assert (X == copy).all()


def powerlaw_mean(x0, xf, alpha):
    a1 = 1 - alpha
    value = -a1 / (x0**a1 - xf**a1)
    value /= alpha - 2
    value *= x0 ** (a1 + 1) - xf ** (a1 + 1)
    return value


def test_snr_sampler():
    sampler = SnrSampler(
        10,
        1,
        100,
        alpha=3,
        decay_steps=2,
    )
    tols = dict(abs_tol=0, rel_tol=0.1)
    vals = sampler(1000)
    assert vals.min().item() > 10
    assert vals.max().item() < 100

    expected_mean = powerlaw_mean(10, 100, 3)
    assert math.isclose(vals.mean().item(), expected_mean, **tols)

    sampler.step()
    vals = sampler(1000)
    assert vals.min().item() > 5.5
    assert vals.max().item() < 100

    expected_mean = powerlaw_mean(5.5, 100, 3)
    assert math.isclose(vals.mean().item(), expected_mean, **tols)

    sampler.step()
    vals = sampler(1000)
    assert vals.min().item() > 1
    assert vals.max().item() < 100

    expected_mean = powerlaw_mean(1, 100, 3)
    assert math.isclose(vals.mean().item(), expected_mean, **tols)

    # verify that an additional step does nothing
    sampler.step()
    vals = sampler(1000)
    assert vals.min().item() > 1
    assert vals.max().item() < 100

    expected_mean = powerlaw_mean(1, 100, 3)
    assert math.isclose(vals.mean().item(), expected_mean, **tols)
