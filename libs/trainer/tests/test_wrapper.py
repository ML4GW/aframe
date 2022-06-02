import os
import sys

import h5py
import numpy as np
import pytest

from bbhnet.data import GlitchSampler, RandomWaveformDataset, WaveformSampler
from bbhnet.trainer.wrapper import trainify


@pytest.fixture(scope="session")
def outdir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("out")
    return out_dir


@pytest.fixture(scope="session")
def data_directory(tmpdir_factory):
    datadir = tmpdir_factory.mktemp("data")
    create_all_data_files(datadir)
    return datadir


def create_data_file(data_dir, filename, dataset_names, data):
    with h5py.File(data_dir.join(filename), "w") as f:
        for name, data in zip(dataset_names, data):
            f.create_dataset(name, data=data)


def create_all_data_files(datadir):

    sample_rate = 2048
    waveform_duration = 4
    signal_length = waveform_duration * sample_rate
    fake_background = np.random.randn(1000 * signal_length)
    fake_glitches = np.random.randn(100, signal_length)
    fake_waveforms = np.random.randn(100, 2, signal_length)

    create_data_file(
        datadir,
        "hanford_background.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )
    create_data_file(
        datadir,
        "livingston_background.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )

    create_data_file(
        datadir,
        "hanford_background_val.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )
    create_data_file(
        datadir,
        "livingston_background_val.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )

    create_data_file(
        datadir,
        "glitches.h5",
        ["H1_glitches", "L1_glitches"],
        [fake_glitches, fake_glitches],
    )
    create_data_file(
        datadir,
        "glitches_val.h5",
        ["H1_glitches", "L1_glitches"],
        [fake_glitches, fake_glitches],
    )

    create_data_file(datadir, "signals.h5", ["signals"], [fake_waveforms])
    create_data_file(datadir, "signals_val.h5", ["signals"], [fake_waveforms])


def return_random_waveform_datasets(
    data_directory: str, outdir: str, **kwargs
):

    train_files = {
        "glitch dataset": os.path.join(data_directory, "glitches.h5"),
        "signal dataset": os.path.join(data_directory, "signals.h5"),
        "hanford background": os.path.join(
            data_directory, "hanford_background.h5"
        ),
        "livingston background": os.path.join(
            data_directory, "livingston_background.h5"
        ),
    }

    val_files = {
        "glitch dataset": os.path.join(data_directory, "glitches_val.h5"),
        "signal dataset": os.path.join(data_directory, "signals.h5"),
        "hanford background": os.path.join(
            data_directory, "hanford_background_val.h5"
        ),
        "livingston background": os.path.join(
            data_directory, "livingston_background_val.h5"
        ),
    }

    min_snr = 4
    max_snr = 100
    highpass = 32
    device = "cpu"
    sample_rate = 2048
    glitch_frac = 0.4
    waveform_frac = 0.4
    batches_per_epoch = 20
    batch_size = 64
    kernel_length = 2
    # initiate training glitch sampler
    train_glitch_sampler = GlitchSampler(
        train_files["glitch dataset"], device=device
    )

    # initiate training waveform sampler
    train_waveform_sampler = WaveformSampler(
        train_files["signal dataset"],
        sample_rate,
        min_snr,
        max_snr,
        highpass,
    )

    # deterministic validation glitch sampler
    # 'determinisitc' key word not yet implemented,
    # just an idea.
    val_glitch_sampler = GlitchSampler(
        val_files["glitch dataset"],
        device=device,
    )

    # deterministic validation waveform sampler
    val_waveform_sampler = WaveformSampler(
        val_files["signal dataset"],
        sample_rate,
        min_snr,
        max_snr,
        highpass,
    )

    # create full training dataloader
    train_dataset = RandomWaveformDataset(
        train_files["hanford background"],
        train_files["livingston background"],
        kernel_length,
        sample_rate,
        batch_size,
        batches_per_epoch,
        train_waveform_sampler,
        waveform_frac,
        train_glitch_sampler,
        glitch_frac,
        device,
    )

    # create full validation dataloader
    valid_dataset = RandomWaveformDataset(
        val_files["hanford background"],
        val_files["livingston background"],
        kernel_length,
        sample_rate,
        batch_size,
        batches_per_epoch,
        val_waveform_sampler,
        waveform_frac,
        val_glitch_sampler,
        glitch_frac,
        device,
    )

    return train_dataset, valid_dataset


def test_wrapper(data_directory, outdir):

    fn = trainify(return_random_waveform_datasets)

    # make sure we can run the function as-is with regular arguments
    train_dataset, valid_dataset = fn(data_directory, outdir)
    assert isinstance(train_dataset, RandomWaveformDataset)
    assert isinstance(valid_dataset, RandomWaveformDataset)

    # call function passing keyword args
    # for train function
    result = fn(
        data_directory,
        outdir,
        max_epochs=1,
        arch="resnet",
        layers=[2, 2, 2],
    )
    assert len(result["train_loss"]) == 1

    sys.argv = [
        None,
        "--data-directory",
        str(data_directory),
        "--outdir",
        str(outdir),
        "--max-epochs",
        "1",
        "resnet",
        "--layers",
        "2",
        "2",
    ]

    # since trainify wraps function w/ typeo
    # looks for args from command line
    # i.e. from sys.argv
    result = fn()
    assert len(result["train_loss"]) == 1
