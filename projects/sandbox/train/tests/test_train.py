import h5py
import pytest
from train.train import main as train


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


def test_train(
    data_dir,
    out_dir,
    log_dir,
    arange_background,
    arange_waveforms,
    arange_glitches,
    sample_rate,
):

    hanford_background_path = arange_background(
        data_dir, sample_rate, 10000, "H1"
    )
    livingston_background_path = arange_background(
        data_dir, sample_rate, 10000, "L1"
    )

    glitch_dataset = arange_glitches(data_dir, sample_rate, 4, 500)
    waveform_dataset = arange_waveforms(data_dir, sample_rate, 4, 500)

    train_dataset, valid_dataset, preprocessor = train(
        hanford_background_path,
        livingston_background_path,
        glitch_dataset,
        waveform_dataset,
        out_dir,
        log_dir,
        glitch_prob=0.25,
        waveform_prob=0.5,
        kernel_length=2,
        sample_rate=sample_rate,
        batch_size=512,
        mean_snr=10,
        std_snr=4,
        min_snr=2,
        highpass=32,
        batches_per_epoch=200,
        # preproc args
        fduration=1,
        trigger_distance=-0.5,
        # validation args
        valid_frac=0.2,
        valid_stride=1,
    )

    with h5py.File(hanford_background_path) as f:
        hanford_background = f["hoft"][()]

    with h5py.File(livingston_background_path) as f:
        livingston_background = f["hoft"][()]

    print(train_dataset.X.shape)
    assert train_dataset.X == [
        hanford_background[: (8000 * sample_rate)],
        livingston_background[: (8000 * sample_rate)],
    ]
