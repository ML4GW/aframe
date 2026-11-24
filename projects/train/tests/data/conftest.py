"""Shared fixtures and helpers for data tests."""

import logging
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from ledger.injections import (
    WaveformPolarizationSet,
    WaveformSet,
    waveform_class_factory,
)

from train.metrics import get_timeslides

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 2048
IFOS = ["H1", "L1"]
KERNEL_LENGTH = 2.0
FDURATION = 1.0
PSD_LENGTH = 4.0
SAMPLE_LENGTH = PSD_LENGTH + KERNEL_LENGTH + FDURATION  # 7.0 s
WAVEFORM_DURATION = 8.0
WAVEFORM_SAMPLES = int(WAVEFORM_DURATION * SAMPLE_RATE)
WAVEFORM_RIGHT_PAD = 0.0
BATCH_SIZE = 8
VALID_LIVETIME = 20.0
VALID_STRIDE = 1.0
MIN_VALID_DURATION = 25.0
BG_DURATION = 30
BG_SAMPLES = BG_DURATION * SAMPLE_RATE  # 3840

# Derived geometry (left_pad=right_pad=0 in hparams)
FILTER_SIZE = int(FDURATION * SAMPLE_RATE)  # 128
LEFT_PAD_SIZE = FILTER_SIZE // 2  # 64
RIGHT_PAD_SIZE = FILTER_SIZE // 2  # 64
# "unwhitened kernel" = kernel_length*sr + fduration*sr
KERNEL_SIZE = int(KERNEL_LENGTH * SAMPLE_RATE) + FILTER_SIZE  # 384
# waveform length after slice_waveforms
SLICED_SAMPLES = 2 * KERNEL_SIZE - LEFT_PAD_SIZE - RIGHT_PAD_SIZE  # 640


# ---------------------------------------------------------------------------
# Low-level file builders
# ---------------------------------------------------------------------------


def _bilby_param_arrays(n: int) -> dict:
    """Return zeroed arrays for every BilbyParameterSet field."""
    names = [
        "mass_1",
        "mass_2",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "ra",
        "dec",
        "redshift",
        "psi",
        "theta_jn",
        "phase",
    ]
    return {
        k: np.ones(n, dtype=np.float64)
        if k in ("mass_1", "mass_2")
        else np.zeros(n, dtype=np.float64)
        for k in names
    }


def make_background_dir(tmp_path: Path) -> Path:
    """Create a minimal background directory with train + val HDF5 files."""
    bg_dir = tmp_path / "background"
    bg_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for stem in ["train-30", "val-30"]:
        with h5py.File(bg_dir / f"{stem}.hdf5", "w") as f:
            for ifo in IFOS:
                data = rng.standard_normal(BG_SAMPLES).astype(np.float32)
                dset = f.create_dataset(ifo, data=data)
                dset.attrs["dx"] = 1.0 / SAMPLE_RATE
    return tmp_path


def make_val_waveform_file(tmp_path: Path, n: int = 20) -> Path:
    """Write a WaveformSet (val) HDF5 file and return its path."""
    T = WAVEFORM_SAMPLES
    rng = np.random.default_rng(1)
    WS = waveform_class_factory(IFOS, WaveformSet, "WaveformSet")
    ifo_fields = {
        ifo.lower(): rng.standard_normal((n, T)).astype(np.float32)
        for ifo in IFOS
    }
    params = _bilby_param_arrays(n)
    wset = WS(
        **ifo_fields,
        **params,
        snr=np.ones(n),
        ifo_snrs=np.ones((n, len(IFOS))),
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        duration=WAVEFORM_DURATION,
        right_pad=WAVEFORM_RIGHT_PAD,
        num_injections=n,
    )
    path = tmp_path / "val_waveforms.hdf5"
    wset.write(path)
    return path


def make_training_waveform_file(tmp_path: Path, n: int = 100) -> Path:
    """Write a WaveformPolarizationSet (training) HDF5 file and return its
    path."""
    T = WAVEFORM_SAMPLES
    rng = np.random.default_rng(2)
    params = _bilby_param_arrays(n)
    wps = WaveformPolarizationSet(
        cross=rng.standard_normal((n, T)).astype(np.float32),
        plus=rng.standard_normal((n, T)).astype(np.float32),
        **params,
        sample_rate=SAMPLE_RATE,
        duration=WAVEFORM_DURATION,
        right_pad=WAVEFORM_RIGHT_PAD,
        num_injections=n,
    )
    wf_dir = tmp_path / "training_waveforms"
    wf_dir.mkdir(exist_ok=True)
    path = wf_dir / "waveforms.hdf5"
    wps.write(path)
    return path


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path):
    """Root temp directory with background/ subfolder populated."""
    return make_background_dir(tmp_path)


@pytest.fixture()
def val_waveform_file(data_dir):
    return make_val_waveform_file(data_dir)


@pytest.fixture()
def training_waveform_file(data_dir):
    return make_training_waveform_file(data_dir)


# ---------------------------------------------------------------------------
# Dataset construction helpers
# ---------------------------------------------------------------------------


def base_dataset_kwargs(data_dir, waveform_sampler) -> dict:
    """Common __init__ kwargs for TimeDomainSupervisedAframeDataset."""
    return {
        "background_dir": str(data_dir),
        "waveforms_dir": str(data_dir),
        "ifos": IFOS,
        "sample_rate": SAMPLE_RATE,
        "dec": torch.distributions.Uniform(-1.0, 1.0),
        "psi": torch.distributions.Uniform(0.0, float(np.pi)),
        "phi": torch.distributions.Uniform(0.0, 2.0 * float(np.pi)),
        "batches_per_epoch": 4,
        "num_files_per_batch": 1,
        "waveform_sampler": waveform_sampler,
        "batch_size": BATCH_SIZE,
        "kernel_length": KERNEL_LENGTH,
        "fduration": FDURATION,
        "psd_length": PSD_LENGTH,
        "waveform_prob": 1.0,
        "left_pad": 0.0,
        "right_pad": 0.0,
        "snr_sampler": torch.distributions.Uniform(8.0, 20.0),
        "valid_stride": VALID_STRIDE,
        "min_valid_duration": MIN_VALID_DURATION,
        "valid_livetime": VALID_LIVETIME,
        "swap_prob": 0.1,
        "mute_prob": 0.05,
    }


def setup_dataset_manually(dataset) -> None:
    """
    Initialise a dataset's runtime attributes without requiring a
    Lightning Trainer (needed for setup() + transforms_to_device()).
    """
    dataset._logger = logging.getLogger("test")
    dataset.train_fnames, dataset.valid_fnames = dataset.train_val_split()

    val_bg = dataset.load_val_background(dataset.valid_fnames)
    dataset.timeslides, dataset.valid_loader_length = get_timeslides(
        val_bg,
        dataset.hparams.valid_livetime,
        dataset.hparams.sample_rate,
        dataset.sample_length,
        dataset.hparams.valid_stride,
        dataset.val_batch_size,
    )
    dataset.val_waveforms, dataset.val_params = (
        dataset.waveform_sampler.get_val_waveforms(1, 0)
    )
    dataset.build_transforms()
    # skip transforms_to_device(): no trainer / GPU required in unit tests


def make_synthetic_X(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Background tensor of the shape expected by inject()."""
    n_samples = int(SAMPLE_LENGTH * SAMPLE_RATE)
    return torch.randn(batch_size, len(IFOS), n_samples)


def make_synthetic_waveforms(
    n: int = 32,
    *,
    sliced: bool,
) -> torch.Tensor:
    """
    Waveform tensor (cross/plus) for inject().

    sliced=True  → pre-sliced to SLICED_SAMPLES (disk-loader mode).
    sliced=False → full WAVEFORM_SAMPLES length (generator mode).
    """
    T = SLICED_SAMPLES if sliced else WAVEFORM_SAMPLES
    return torch.randn(n, 2, T)
