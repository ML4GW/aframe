import h5py
import numpy as np
import pytest
from train.train import main as train


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=[0.2])
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
    valid_frac,
):

    duration = 10000
    num_ifos = 2
    kernel_length = 2
    fduration = 1

    hanford_background_path = arange_background(
        data_dir, sample_rate, duration, "H1"
    )
    livingston_background_path = arange_background(
        data_dir, sample_rate, duration, "L1"
    )

    glitch_dataset = arange_glitches(data_dir, sample_rate, 4, 500)
    waveform_dataset = arange_waveforms(data_dir, sample_rate, 4, 500)

    train_dataset, validator, preprocessor = train(
        hanford_background_path,
        livingston_background_path,
        glitch_dataset,
        waveform_dataset,
        out_dir,
        log_dir,
        glitch_prob=0.25,
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

    # train_background = np.stack((hanford_background, livingston_background))[
    #     :, :int(duration * (1 - valid_frac) * sample_rate - 1)
    # ]
    # assert train_dataset.X.numpy() == pytest.approx(train_background)

    num_kernels = (duration*valid_frac) // (kernel_length - fduration) - 1
    num_kernels = int(num_kernels)
    kernel_size = kernel_length*sample_rate
    assert (validator.background_loader.dataset[:][0].numpy().shape == np.array(
        (num_kernels, num_ifos, kernel_size)
    )).all()
    assert validator.glitch_loader.dataset[:][0].shape == [100, 2, 2048]
    assert validator.signal_loader.dataset[:][0].shape == [100, 2, 2048]
    # assert pytest.approx(train_dataset.X, [
    #     hanford_background[:8000*sample_rate],
    #     livingston_background[:8000*sample_rate]
    # ])

    # assert validator.background_loader.dataset == [
    #     hanford_background[8000*sample_rate:],
    #     livingston_background[8000*sample_rate:]
    # ]
