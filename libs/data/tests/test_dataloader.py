import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gwpy.timeseries import TimeSeries

from bbhnet.data import dataloader
from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.waveform_sampler import WaveformSampler


@pytest.fixture
def data_length():
    return 128


@pytest.fixture(params=[0.01, 0.1, 0.5, 0.9])
def glitch_frac(request):
    return request.param


@pytest.fixture(params=[0.01, 0.1, 0.5, 0.9])
def waveform_frac(request):
    return request.param


@pytest.fixture
def t0():
    return 1234567890


@pytest.fixture
def data_size(sample_rate, data_length):
    return int(sample_rate * data_length)


@pytest.fixture(scope="function")
def random_data(data_size):
    return np.random.randn(data_size)


@pytest.fixture
def zero_data(data_size):
    return np.zeros((data_size,))


@pytest.fixture
def sequential_data(data_size):
    return np.arange(data_size)


@pytest.fixture
def write_background(write_timeseries, t0, data_dir):
    def f(fname, x):
        write_timeseries(fname, hoft=x, t0=t0)
        return data_dir / fname

    return f


@pytest.fixture
def random_hanford_background(random_data, write_background):
    return write_background("hanford.h5", random_data)


@pytest.fixture
def random_livingston_background(random_data, write_background):
    return write_background("livingston.h5", random_data)


@pytest.fixture
def sequential_hanford_background(sequential_data, write_background):
    return write_background("hanford.h5", sequential_data)


@pytest.fixture
def sequential_livingston_background(sequential_data, write_background):
    return write_background("livingston.h5", -sequential_data)


@pytest.fixture
def zeros_hanford_background(zero_data, write_background):
    return write_background("hanford.h5", zero_data)


@pytest.fixture
def zeros_livingston_background(zero_data, write_background):
    return write_background("livingston.h5", zero_data)


@pytest.fixture(params=["path", "sampler"])
def glitch_sampler(arange_glitches, request, device):
    if request.param == "path":
        return arange_glitches
    else:
        return GlitchSampler(arange_glitches, device)


def validate_sequential(X):
    # make sure that we're not sampling in order
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    assert not (np.diff(X[:, 0, 0]) == 1).all()

    # now make sure each timeseries is sequential
    # and that the two interferometers don't match
    for x in X:
        assert not (x[0] == x[1]).all()
        for i, x_ifo in enumerate(x):
            diff = (-1) ** i
            assert (np.diff(x_ifo) == diff).all()


@patch("bbhnet.data.RandomWaveformDataset.whiten", new=lambda obj, x: x)
def test_random_waveform_dataset(
    sequential_hanford_background,
    sequential_livingston_background,
    sample_rate,
    device,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        sequential_hanford_background,
        sequential_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        device=device,
    )

    # test the background sampling method to make sure
    # that the base batch is generated properly
    X = dataset.sample_from_background().cpu().numpy()
    validate_sequential(X)

    # now go through and make sure that the iteration
    # method generates data of the right size
    for i, (X, y) in enumerate(dataset):
        assert X.shape == (batch_size, 2, sample_rate)
        validate_sequential(X)

        # make sure targets are all 0 because we
        # have no waveform sampling
        assert y.shape == (batch_size,)
        assert not y.cpu().numpy().any()

    assert i == 9


def test_random_waveform_dataset_whitening(
    random_hanford_background,
    random_livingston_background,
    sample_rate,
    data_length,
    device,
):
    """
    Test the `.whiten` method to make sure that it
    produces results roughly consistent with gwpy's
    whitening functionality using the background
    data's ASD
    """

    # create a dataset from the background
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        random_hanford_background,
        random_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        device=device,
    )

    # whiten a random batch of data manually
    X = np.random.randn(32, 2, sample_rate)
    whitened = dataset.whiten(torch.Tensor(X).to(device)).cpu().numpy()

    # for each sample, whiten the sample using the
    # corresponding background ASD with gwpy and
    # ensure that the results are reasonably close
    errs = []
    for x, w in zip(X, whitened):
        for i, bkgrd in enumerate(
            [dataset.hanford_background, dataset.livingston_background]
        ):
            bkgrd_ts = TimeSeries(bkgrd.cpu().numpy(), dt=1 / sample_rate)
            bkgrd_asd = bkgrd_ts.asd(fftlength=2)
            ts = TimeSeries(x[i], dt=1 / sample_rate)
            ts = ts.whiten(asd=bkgrd_asd).value

            errs.extend(np.abs(ts - w[i]) / np.abs(ts))

    # make sure that the relative error from the two
    # whitening methods is in the realm of the reasonable.
    # We could make these bounds tighter for most sample
    # rates of interest, but the 512 just has too much noise
    assert np.percentile(errs, 90) < 0.001
    assert np.percentile(errs, 99) < 0.01


@patch("bbhnet.data.RandomWaveformDataset.whiten", new=lambda obj, x: x)
def test_glitch_sampling(
    random_hanford_background,
    random_livingston_background,
    glitch_sampler,
    glitch_frac,
    sample_rate,
    device,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        random_hanford_background,
        random_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        glitch_frac=glitch_frac,
        glitch_sampler=glitch_sampler,
        batches_per_epoch=10,
        device=device,
    )
    expected_num = max(1, int(glitch_frac * batch_size))
    assert dataset.num_glitches == expected_num

    for X, y in dataset:
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        assert not y.any()

        for i, x in enumerate(X):
            # check if either ifo channel is conti
            is_glitch = (np.abs(np.diff(x, axis=-1)) == 1).all(axis=-1)

            # check that either this sample is one of the
            # first `num_glitches` in the batch or does not
            # have a glitch
            assert (i < dataset.num_glitches) ^ (not is_glitch.any())

    if device == "cpu":
        return

    dataset.batches_per_epoch = 100
    start_time = time.time()
    for _ in dataset:
        continue
    end_time = time.time()
    assert ((end_time - start_time) / 100) < 0.05


def test_waveform_sampling(
    random_hanford_background,
    random_livingston_background,
    sine_waveforms,
    waveform_frac,
    sample_rate,
    device,
):
    waveform_sampler = WaveformSampler(
        sine_waveforms, sample_rate, min_snr=20, max_snr=40
    )

    # initialize dataset with random data so
    # that we have a valid whitening filter,
    # but zero the background out later so that
    # we can check the waveforms
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        random_hanford_background,
        random_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        waveform_sampler=waveform_sampler,
        waveform_frac=waveform_frac,
        batches_per_epoch=10,
        device=device,
    )
    expected_num = max(1, int(waveform_frac * batch_size))
    assert dataset.num_waveforms == expected_num

    dataset.hanford_background *= 0
    dataset.livingston_background *= 0

    if dataset.num_waveforms > 10:
        with pytest.raises(ValueError):
            next(iter(dataset))
        return

    for X, y in dataset:
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        for i, x in enumerate(X):
            # check to make sure ifo is not all 0s
            is_background = (x == 0).all(axis=-1)

            # check that either this sample is one of the
            # first `num_glitches` in the batch or does not
            # have a glitch
            limit = batch_size - dataset.num_waveforms
            assert (i >= limit) ^ (is_background.all())
            assert y[i] == int(not is_background.all())

    if device == "cpu":
        return

    dataset.batches_per_epoch = 100
    start_time = time.time()
    for _ in dataset:
        continue
    end_time = time.time()
    assert ((end_time - start_time) / 100) < 0.05
