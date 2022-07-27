import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gwpy.frequencyseries import FrequencySeries

from bbhnet.data import dataloader
from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.waveform_sampler import WaveformSampler


def mock_asd(data_length, sample_rate):
    # mock asd object to ones
    # so that the ValueError doesn't
    # get raised  for 0 asd values
    df = 1 / data_length
    fmax = sample_rate / 2
    nfreqs = int(fmax / df)
    asd = FrequencySeries(np.ones(nfreqs), df=df, channel="H1:STRAIN")
    return asd


@pytest.fixture
def data_length():
    return 128


@pytest.fixture
def kernel_length():
    return 1


@pytest.fixture(params=[8, 32])
def batch_size(request):
    return request.param


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


@pytest.fixture
def ones_data(data_size):
    return np.ones((data_size,))


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
def sequential_hanford_background(sequential_data, write_background):
    return write_background("hanford.h5", sequential_data)


@pytest.fixture
def sequential_livingston_background(sequential_data, write_background):
    return write_background("livingston.h5", -sequential_data)


@pytest.fixture
def ones_hanford_background(ones_data, write_background):
    return write_background("hanford.h5", ones_data)


@pytest.fixture
def ones_livingston_background(ones_data, write_background):
    return write_background("livingston.h5", ones_data)


@pytest.fixture(params=["path", "sampler"])
def glitch_sampler(arange_glitches, request):
    if request.param == "path":
        return arange_glitches
    else:
        return GlitchSampler(arange_glitches)


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


def validate_speed(dataset, N, limit):
    dataset.batches_per_epoch = N
    start_time = time.time()
    for _ in dataset:
        continue
    delta = time.time() - start_time
    assert delta / N < limit


def validate_dataset(dataset, cutoff_idx, target):
    # TODO: include "test_sample: Callable" argument
    # for testing each individual X, y sample that
    # isn't background. Check that the center is in
    # and for sines check that the wave has a valid freq
    for X, y in dataset:
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        for i, x in enumerate(X):
            # check to make sure ifo is not all 0s
            is_background = (x == 1).all()

            if target:
                # y == 1 targets should come at the end
                not_background = i >= cutoff_idx
                assert y[i] == int(not is_background)
            else:
                # y == 0 targets that _aren't_ background
                # should come at the start
                not_background = i < cutoff_idx

            # make sure you're either in the position
            # expected for a non-background of the
            # indicated target type, or you're background
            assert not_background ^ is_background, (i, x)


# Test the training dataset class first
def test_random_waveform_dataset(
    sequential_hanford_background,
    sequential_livingston_background,
    sample_rate,
    data_size,
    device,
    frac,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        sequential_hanford_background,
        sequential_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        frac=frac,
    )

    if frac is not None:
        data_size = int(abs(frac) * data_size)

    for ifo in ["hanford", "livingston"]:
        background = getattr(dataset, f"{ifo}_background")
        assert background.device.type == "cpu"
        assert background.dtype == torch.float64
        assert len(background) == data_size

    dataset.to(device)
    for ifo in ["hanford", "livingston"]:
        background = getattr(dataset, f"{ifo}_background")
        assert background.device.type == device
        assert background.dtype == torch.float32

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


def test_random_waveform_dataset_with_glitch_sampling(
    ones_hanford_background,
    ones_livingston_background,
    glitch_sampler,
    glitch_frac,
    sample_rate,
    data_length,
    device,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        ones_hanford_background,
        ones_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        glitch_frac=glitch_frac,
        glitch_sampler=glitch_sampler,
        batches_per_epoch=10,
    )
    dataset.to(device)
    assert dataset.glitch_sampler.hanford.device.type == device
    assert dataset.glitch_sampler.livingston.device.type == device

    expected_num = max(1, int(glitch_frac * batch_size))
    assert dataset.num_glitches == expected_num
    validate_dataset(dataset, dataset.num_glitches, 0)

    if device == "cpu":
        return
    validate_speed(dataset, N=100, limit=0.05)


def test_random_waveform_dataset_with_waveform_sampling(
    ones_hanford_background,
    ones_livingston_background,
    sine_waveforms,
    waveform_frac,
    sample_rate,
    data_length,
    device,
):
    waveform_sampler = WaveformSampler(
        sine_waveforms, sample_rate, min_snr=20, max_snr=40
    )

    batch_size = 32
    with patch(
        "gwpy.timeseries.TimeSeries.asd",
        return_value=mock_asd(data_length, sample_rate),
    ) as mock:
        dataset = dataloader.RandomWaveformDataset(
            ones_hanford_background,
            ones_livingston_background,
            kernel_length=1,
            sample_rate=sample_rate,
            batch_size=batch_size,
            waveform_sampler=waveform_sampler,
            waveform_frac=waveform_frac,
            batches_per_epoch=10,
        )
        dataset.to(device)

    # TODO: test that we don't need to be fit
    # if the waveform sampler has already been fit
    mock.assert_called()
    expected_num = max(1, int(waveform_frac * batch_size))
    assert dataset.num_waveforms == expected_num

    # if the dataset is going to request more waveforms
    # than we have, a ValueError should get raised
    # TODO: now that we're using more waveforms, this won't
    # get raised, so check this at the bottom with a dataloader
    # that requests a lot of waveforms, or with a smaller sampler
    # if dataset.num_waveforms > 10:
    #     with pytest.raises(ValueError):
    #         next(iter(dataset))
    #     return
    validate_dataset(dataset, batch_size - dataset.num_waveforms, 1)

    if device == "cpu":
        return
    validate_speed(dataset, N=100, limit=0.05)


# now run the same tests for the deterministic validation sampler
@pytest.fixture(params=[0.1, 0.5, 1])
def stride(request):
    return request.param


def test_deterministic_waveform_dataset(
    sequential_hanford_background,
    sequential_livingston_background,
    data_size,
    stride,
    sample_rate,
    device,
    frac,
):
    batch_size = 32
    kernel_length = 1
    dataset = dataloader.DeterministicWaveformDataset(
        sequential_hanford_background,
        sequential_livingston_background,
        kernel_length=kernel_length,
        stride=stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        frac=frac,
    )

    assert dataset.waveforms is None
    assert dataset.glitches is None
    assert dataset.background.device.type == "cpu"
    assert dataset.background.dtype == torch.float64

    if frac is not None:
        data_size = int(abs(frac) * data_size)
    assert dataset.background.shape[-1] == data_size
    first_idx = dataset.background[0, 0]

    dataset.to(device)
    assert dataset.background.device.type == device
    assert dataset.background.dtype == torch.float32

    stride_size = int(stride * sample_rate)
    kernel_size = int(kernel_length * sample_rate)

    num_kernels = (data_size - kernel_size) // stride_size + 1
    num_batches, leftover = divmod(num_kernels, batch_size)

    for i, (X, y) in enumerate(dataset):
        assert (y.cpu().numpy() == 0).all()

        X = X.cpu().numpy()
        if i == num_batches and leftover > 0:
            expected_batch = leftover
        else:
            expected_batch = batch_size
        assert X.shape == (expected_batch, 2, sample_rate)

        for j, x in enumerate(X):
            start = first_idx + stride_size * (i * batch_size + j)
            stop = start + kernel_size
            expected = np.arange(start, stop)

            for k, ifo in enumerate(x):
                power = (-1) ** k
                assert (ifo == power * expected).all()

    assert i == (num_batches if leftover > 0 else num_batches - 1)


@pytest.fixture
def num_non_background():
    return 500


@pytest.fixture
def validate_deterministic_dataset(
    data_size, kernel_length, sample_rate, stride, num_non_background, frac
):
    if frac is not None:
        data_size = int(abs(frac) * data_size)

    kernel_size = int(kernel_length * sample_rate)
    stride_size = int(stride * sample_rate)
    num_kernels = (data_size - kernel_size) // stride_size + 1

    def func(dataset, batch_size, target):
        num_batches = (num_kernels - 1) // batch_size + 1
        last_batch = num_kernels % batch_size or batch_size

        if target == 0 and last_batch < 2:
            dataset.background = dataset.background[:, :-stride_size]
            num_batches -= 1
            actual_kernels = num_kernels - 1
            last_batch = batch_size
        else:
            actual_kernels = num_kernels

        factor = 2 - target
        nb_batch = batch_size // factor
        num_nb_batches = (num_non_background - 1) // nb_batch + 1
        nb_last_batch = num_non_background % nb_batch or nb_batch

        for i, (X, y) in enumerate(dataset):
            iteration, idx = divmod(i, num_batches)
            y_exp = 0 if iteration == 0 else target

            assert (y.cpu().numpy() == y_exp).all()
            X = X.cpu().numpy()

            # make sure that our batch_size is as expected
            expected_batch = batch_size

            end_of_background = idx == (num_batches - 1)
            end_of_nb = (i - num_batches) == (num_nb_batches - 1)
            if end_of_background:
                expected_batch = last_batch

                if iteration > 0 and (
                    not end_of_nb or last_batch < nb_last_batch
                ):
                    if target == 0 and last_batch % 2 != 0:
                        expected_batch -= 1

                    # we ran out of background data before we got
                    # through all of our "events." So if the corresponding
                    # batch is too short, it will tack more events on to
                    # the last batch, so record that here
                    end_of_nb = False
                    nb_last_batch += nb_batch - last_batch // factor
                    if nb_last_batch >= nb_batch:
                        # if we added enough to add a whole new batch,
                        # increment our expected number of batches
                        num_nb_batches += 1
                        nb_last_batch -= nb_batch

            if iteration > 0 and end_of_nb:
                expected_batch = nb_last_batch * factor

            msg = f"{i}, {iteration}, {idx}, {num_batches}, {num_nb_batches}"
            assert X.shape == (expected_batch, 2, sample_rate), msg

            if iteration == 0:
                assert (X == 1).all(), msg
            elif target == 1:
                # we're dealing with waveforms
                expected = np.arange(X[0, 0, 0], X[-1, 0, 0] + 1) - 1
                assert (X[:, 0, 0] == (1 + expected)).all(), msg
                assert (X[:, 0, -1] == (1 + expected)).all()
                assert (X[:, 1, 0] == (1 - expected)).all()
            elif target == 0:
                # these are glitches, make sure they got
                # inserted non-coincidentally
                expected = np.arange(
                    X[0, 0, 0], X[expected_batch // 2 - 1, 0, 0] + 1
                )
                assert (X[: expected_batch // 2, 0, 0] == expected).all(), msg
                assert (X[: expected_batch // 2, 1, 0] == 1).all(), msg

                assert (X[expected_batch // 2 :, 1, 0] == -expected).all(), msg
                assert (X[expected_batch // 2 :, 0, 0] == 1).all(), msg

        if target == 0:
            # TODO: having trouble working out some of the algebra
            # for the expected behavior of glitches at the edge
            # of the dataset so skipping the expected number of
            # iterations check for now
            return

        # the plus two is to account for the fact that
        # we'll have one iteration for the background.
        # We could just add one and save ourselves the
        # subtraction in the next line, but this makes
        # the expected behavior very explicit
        expected_its = (num_non_background * factor - 1) // actual_kernels + 2
        assert iteration == (expected_its - 1)

    return func


@pytest.fixture
def events(kernel_length, sample_rate, num_non_background):
    kernel_size = int(sample_rate * kernel_length)
    waveforms = np.ones((kernel_size, num_non_background))
    waveforms += np.arange(num_non_background)
    events = waveforms.T
    return [events, -events]


def test_deterministic_waveform_dataset_with_glitch_sampling(
    ones_hanford_background,
    ones_livingston_background,
    batch_size,
    sample_rate,
    data_length,
    kernel_length,
    stride,
    device,
    events,
    frac,
    validate_deterministic_dataset,
):
    sampler = MagicMock()
    sampler.sample = MagicMock(return_value=events)
    dataset = dataloader.DeterministicWaveformDataset(
        ones_hanford_background,
        ones_livingston_background,
        kernel_length=kernel_length,
        stride=stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        glitch_sampler=sampler,
        frac=frac,
    )
    assert dataset.glitches is not None
    assert dataset.glitches.device.type == "cpu"

    dataset.to(device)
    assert dataset.glitches.device.type == device

    validate_deterministic_dataset(dataset, batch_size, target=0)


def test_deterministic_waveform_dataset_with_waveform_sampling(
    ones_hanford_background,
    ones_livingston_background,
    batch_size,
    sample_rate,
    kernel_length,
    stride,
    device,
    events,
    frac,
    validate_deterministic_dataset,
):
    waveforms = np.stack(events).transpose(1, 0, 2)
    sampler = MagicMock()
    sampler.sample = MagicMock(return_value=waveforms)
    dataset = dataloader.DeterministicWaveformDataset(
        ones_hanford_background,
        ones_livingston_background,
        kernel_length=kernel_length,
        stride=stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        waveform_sampler=sampler,
        frac=frac,
    )

    assert dataset.waveforms is not None
    assert dataset.waveforms.device.type == "cpu"

    dataset.to(device)
    assert dataset.waveforms.device.type == device

    validate_deterministic_dataset(dataset, batch_size, target=1)
