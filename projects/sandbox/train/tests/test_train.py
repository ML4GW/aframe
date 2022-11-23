import h5py
import numpy as np
import pytest
from train.train import main as train


@pytest.fixture(params=[0.25, 0.67])
def glitch_prob(request):
    return request.param


@pytest.fixture(params=[1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[0.2, 0.6])
def valid_frac(request):
    return request.param


def test_train(
    data_dir,
    out_dir,
    log_dir,
    arange_background,
    arange_waveforms,
    arange_glitches,
    sample_rate,
    glitch_prob,
    valid_frac,
):

    duration = 10000
    num_ifos = 2
    kernel_length = 2
    fduration = 1
    num_glitches = 500
    num_waveforms = 500

    hanford_background_path = arange_background(
        data_dir, sample_rate, duration, "H1"
    )
    livingston_background_path = arange_background(
        data_dir, sample_rate, duration, "L1"
    )

    glitch_dataset = arange_glitches(data_dir, sample_rate, 4, num_glitches)
    waveform_dataset = arange_waveforms(
        data_dir, sample_rate, 4, num_waveforms
    )

    train_dataset, validator, preprocessor = train(
        hanford_background_path,
        livingston_background_path,
        glitch_dataset,
        waveform_dataset,
        out_dir,
        log_dir,
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

    with h5py.File(hanford_background_path) as f:
        hanford_background = f["hoft"][:]

    with h5py.File(livingston_background_path) as f:
        livingston_background = f["hoft"][:]

    # Check that the training background is what it should be
    train_background = np.stack((hanford_background, livingston_background))[
        :, : int(duration * (1 - valid_frac) * sample_rate - 1)
    ]
    assert train_dataset.X.numpy() == pytest.approx(train_background)

    # Check that the shapes of the DataLoaders in validator are the right shape
    num_kernels = (duration * valid_frac) // (kernel_length - fduration) - 1
    num_kernels = int(num_kernels)
    kernel_size = kernel_length * sample_rate

    assert (
        validator.background_loader.dataset[:][0].numpy().shape
        == np.array((num_kernels, num_ifos, kernel_size))
    ).all()

    # This is set up to avoid any off by 1 errors that come from rounding
    total_valid_glitches = 2 * num_glitches - int(
        (1 - valid_frac) * 2 * num_glitches
    )
    num_coinc = int(
        glitch_prob**2 * total_valid_glitches / (1 + glitch_prob**2)
    )
    num_valid_glitches = total_valid_glitches - num_coinc
    assert (
        validator.glitch_loader.dataset[:][0].numpy().shape
        == np.array((num_valid_glitches, num_ifos, kernel_size))
    ).all()

    assert (
        validator.signal_loader.dataset[:][0].numpy().shape
        == np.array((int(valid_frac * num_waveforms), num_ifos, kernel_size))
    ).all()

    # Check that the whitening filter is the correct shape
    assert (
        preprocessor.whitener.time_domain_filter.numpy().shape
        == np.array((num_ifos, 1, fduration * sample_rate - 1))
    ).all()
