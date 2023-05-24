from unittest.mock import MagicMock, patch

import pytest
import torch
from train.data_structures import BBHNetBatchAugmentor


@pytest.fixture
def sample_rate():
    return 512


rand_value = 0.1 + 0.5 * (torch.arange(32) % 2)


@patch("torch.rand", return_value=rand_value)
def test_bbhnet_batch_augmentor():

    augmentor = BBHNetBatchAugmentor(
        ifos=["H1", "L1"],
        sample_rate=2048,
        signal_prob=0.5,
        glitch_sampler=MagicMock(),
        dec=MagicMock(),
        psi=MagicMock(),
        phi=MagicMock(),
        trigger_distance=-0.5,
        plus=torch.zeros((1, 128 * 2)),
        cross=torch.zeros((1, 128 * 2)),
    )

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))

    X, y = augmentor(X, y)
    assert (X[::2] == 1).all().item()
    assert (X[1::2] == 0).all().item()
    assert (y[::2] == 1).all().item()
    assert (y[1::2] == 0).all().item()
