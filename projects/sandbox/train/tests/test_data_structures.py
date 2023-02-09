from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from train.data_structures import (
    BBHInMemoryDataset,
    BBHNetWaveformInjection,
    GlitchSampler,
)


@pytest.fixture
def sample_rate():
    return 512


@pytest.fixture(params=[1, 4])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[8, 32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 10])
def batches_per_epoch(request):
    return request.param


@pytest.fixture(params=[1, 2])
def num_ifos(request):
    return request.param


@pytest.fixture
def sequential_data(num_ifos, sample_rate):
    x = np.arange(num_ifos * 128 * sample_rate)
    x = x.reshape(num_ifos, 128 * sample_rate)
    return x


@pytest.fixture(params=[True, False])
def coincident(request):
    return request.param


def test_bbhnet_in_memory_dataloader(
    sequential_data,
    kernel_length,
    batch_size,
    batches_per_epoch,
    coincident,
    num_ifos,
    sample_rate,
):
    kernel_size = int(kernel_length * sample_rate)
    dataset = BBHInMemoryDataset(
        sequential_data,
        kernel_size,
        batch_size,
        batches_per_epoch=batches_per_epoch,
        preprocessor=None,
        coincident=coincident,
        shuffle=True,
        device="cpu",
    )

    for i, (X, y) in enumerate(dataset):
        assert X.shape == (batch_size, num_ifos, kernel_size)
        assert y.shape == (batch_size, 1)
        assert (y == 0).all().item()

        for sample in X:
            start = sample[0, 0]
            expected = torch.arange(start, start + kernel_size)
            assert (sample[0] == expected).all().item()
            if num_ifos == 1:
                continue

            if coincident:
                expected = expected + 128 * sample_rate
                assert (sample[1] == expected).all().item()
            else:
                start = sample[1, 0]
                expected = torch.arange(start, start + kernel_size)
                assert (sample[1] == expected).all().item()
    assert (i + 1) == batches_per_epoch


def test_bbhnet_in_memory_dataloader_with_preprocessor(
    sequential_data, sample_rate
):
    def preprocessor(X, y):
        X[:, 0] *= 2
        y[::2] = 1
        return X, y

    dataset = BBHInMemoryDataset(
        sequential_data,
        kernel_size=2 * sample_rate,
        batch_size=8,
        batches_per_epoch=2,
        preprocessor=preprocessor,
        coincident=True,
        shuffle=True,
        device="cpu",
    )
    for X, y in dataset:
        assert ((X[:, 0] % 2) == 0).all().item()
        if X.shape[1] > 1:
            assert not ((X[:, 1] % 2) == 0).all().item()

        assert (y[::2] == 1).all().item()
        assert (y[1::2] == 0).all().item()


@pytest.fixture(params=[0, 0.25, 1])
def prob(request):
    return request.param


def get_rand_patch(size):
    probs = torch.linspace(0, 0.99, size[1])
    return torch.stack([probs] * size[0])


def test_glitch_sampler(sample_rate, offset, device, prob):
    glitches = torch.arange(512 * 2 * 10, dtype=torch.float32)
    glitches = glitches.reshape(10, 512 * 2) + 1
    sampler = GlitchSampler(
        prob=prob,
        max_offset=offset,
        h1=glitches,
        l1=-1 * glitches[:9],
    )
    sampler.to(device)
    for glitch in sampler.buffers():
        assert glitch.device.type == device

    X = torch.zeros((8, 2, 512), dtype=torch.float32).to(device)
    probs = get_rand_patch((2, 8)).to(device)
    with patch("torch.rand", return_value=probs):
        inserted, _ = sampler(X, None)

        # TODO: the tests could be more extensive, but
        # then are we functionally just testing sample_kernels?
        if prob == 0:
            assert (inserted == 0).all().item()
        elif prob == 1:
            assert (inserted != 0).all().item()
        else:
            assert (inserted[:2] != 0).all().item()
            assert (inserted[2:] == 0).all().item()


def sample(obj, N):
    return torch.ones((N, 2, 128 * 2)), torch.ones((N, 3))


rand_value = 0.1 + 0.5 * (torch.arange(32) % 2)


@patch("ml4gw.transforms.injection.RandomWaveformInjection.sample", new=sample)
@patch("torch.rand", return_value=rand_value)
def test_bbhnet_waveform_injection(rand_mock):
    tform = BBHNetWaveformInjection(
        sample_rate=128,
        ifos=["H1", "L1"],
        dec=MagicMock(),
        psi=MagicMock(),
        phi=MagicMock(),
        prob=0.5,
        plus=torch.zeros((1, 128 * 2)),
        cross=torch.zeros((1, 128 * 2)),
    )

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))

    X, y = tform(X, y)
    assert (X[::2] == 1).all().item()
    assert (X[1::2] == 0).all().item()
    assert (y[::2] == 1).all().item()
    assert (y[1::2] == 0).all().item()
