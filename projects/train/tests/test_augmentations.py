from unittest.mock import patch

import numpy as np
import pytest
import torch

from train.augmentations import ChannelSwapper


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
