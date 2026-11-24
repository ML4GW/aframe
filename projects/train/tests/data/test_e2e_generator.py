"""
End-to-end integration test — on-the-fly generator path.

Wires TimeDomainSupervisedAframeDataset + a minimal concrete WaveformGenerator
and calls inject() with synthetic background and full-length waveforms
(slice_waveforms is called inside inject() in generator mode).
"""

import pytest
import torch

from train.data.supervised.time_domain import TimeDomainSupervisedAframeDataset
from train.data.waveforms.generator.generator import WaveformGenerator

from conftest import (
    BATCH_SIZE,
    IFOS,
    SAMPLE_RATE,
    WAVEFORM_SAMPLES,
    base_dataset_kwargs,
    make_synthetic_X,
    make_synthetic_waveforms,
    setup_dataset_manually,
)

EXTRINSIC_KEYS = {"dec", "psi", "phi", "snr"}


class _TrivialGenerator(WaveformGenerator):
    """Concrete WaveformGenerator whose forward() returns zeros."""

    def forward(self, **params):
        n = len(next(iter(params.values())))
        return torch.zeros(n, 2, WAVEFORM_SAMPLES)


@pytest.fixture()
def generator_dataset(data_dir, val_waveform_file):
    mock_prior = lambda n, device=None: {  # noqa: E731
        "mass_1": torch.ones(n),
        "mass_2": torch.ones(n),
    }
    sampler = _TrivialGenerator(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        val_waveform_file=val_waveform_file,
        training_prior=mock_prior,
    )
    dataset = TimeDomainSupervisedAframeDataset(
        **base_dataset_kwargs(data_dir, sampler)
    )
    setup_dataset_manually(dataset)
    return dataset


@pytest.fixture()
def inject_outputs(generator_dataset):
    X = make_synthetic_X()
    waveforms = make_synthetic_waveforms(n=32, sliced=False)
    params = {"mass_1": torch.arange(32, dtype=torch.float32)}
    return generator_dataset.inject(X=X, waveforms=waveforms, params=params)


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


def test_no_state_between_calls(generator_dataset):
    X = make_synthetic_X()
    waveforms = make_synthetic_waveforms(n=32, sliced=False)
    params = {"mass_1": torch.ones(32)}

    results = []
    for _ in range(2):
        out = generator_dataset.inject(
            X=X.clone(), waveforms=waveforms.clone(), params=params
        )
        results.append(out)

    for call_idx, (_, _, params_out) in enumerate(results):
        for key in EXTRINSIC_KEYS:
            assert key in params_out, f"Call {call_idx}: missing key '{key}'"
        for key, val in params_out.items():
            assert val.shape == (BATCH_SIZE,), (
                f"Call {call_idx}: params_out['{key}'] shape"
                f" changed to {val.shape}"
            )
