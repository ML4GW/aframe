"""
End-to-end integration test — disk-loader path.

Wires TimeDomainSupervisedAframeDataset + WaveformLoader and calls inject()
with synthetic background and pre-sliced waveforms, asserting that parameter
dicts are correctly shaped, keyed, and have NaN exactly where labels == 0.
"""

import pytest
import torch

from train.data.supervised.time_domain import TimeDomainSupervisedAframeDataset
from train.data.waveforms.loader import WaveformLoader

from conftest import (
    BATCH_SIZE,
    IFOS,
    SAMPLE_RATE,
    base_dataset_kwargs,
    make_synthetic_X,
    make_synthetic_waveforms,
    setup_dataset_manually,
)

EXTRINSIC_KEYS = {"dec", "psi", "phi", "snr"}


@pytest.fixture()
def disk_loader_dataset(data_dir, val_waveform_file, training_waveform_file):
    sampler = WaveformLoader(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        val_waveform_file=val_waveform_file,
        training_waveform_path=training_waveform_file,
    )
    dataset = TimeDomainSupervisedAframeDataset(
        **base_dataset_kwargs(data_dir, sampler)
    )
    setup_dataset_manually(dataset)
    return dataset


@pytest.fixture()
def inject_outputs(disk_loader_dataset):
    """
    Run one inject() call and return (X_out, y, params_out).

    Waveforms are pre-sliced (SLICED_SAMPLES) to match what
    on_before_batch_transfer would produce for the disk-loader path.
    """
    X = make_synthetic_X()
    waveforms = make_synthetic_waveforms(n=32, sliced=True)
    params = {"mass_1": torch.arange(32, dtype=torch.float32)}
    return disk_loader_dataset.inject(X=X, waveforms=waveforms, params=params)


def test_param_keys_and_shape(inject_outputs):
    _, _, params_out = inject_outputs
    for key in EXTRINSIC_KEYS:
        assert key in params_out, f"Missing expected key '{key}'"
    for key, val in params_out.items():
        assert val.shape == (BATCH_SIZE,), (
            f"params_out['{key}'] has shape {val.shape},"
            f" expected ({BATCH_SIZE},)"
        )


def test_nan_matches_labels(inject_outputs):
    _, y, params_out = inject_outputs
    not_injected = y.squeeze() == 0
    for key, val in params_out.items():
        nan_mask = torch.isnan(val)
        assert torch.equal(nan_mask, not_injected), (
            f"NaN mask for '{key}' does not match label==0 mask"
        )


def test_val_params_shape(disk_loader_dataset):
    from train.data.base import _SignalDataset

    val_waveforms = disk_loader_dataset.val_waveforms
    val_params = disk_loader_dataset.val_params

    sig_dset = _SignalDataset(val_waveforms, val_params)
    loader = torch.utils.data.DataLoader(sig_dset, batch_size=4, shuffle=False)
    batch_wf, batch_params = next(iter(loader))

    assert batch_wf.shape[0] == 4
    for key, val in batch_params.items():
        assert val.shape[0] == batch_wf.shape[0], (
            f"params['{key}'] batch dim {val.shape[0]} != waveform batch dim"
        )
