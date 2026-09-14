"""
Unit tests for WaveformGenerator.sample.
"""

import torch

from train.data.waveforms.generator.generator import WaveformGenerator

from conftest import (
    IFOS,
    SAMPLE_RATE,
    WAVEFORM_SAMPLES,
    make_val_waveform_file,
)

import pytest

N = 12
T = WAVEFORM_SAMPLES
NUM_CHANNELS = 2  # cross + plus


class _ConcreteGenerator(WaveformGenerator):
    """Minimal WaveformGenerator with a working forward()."""

    def forward(self, **params):
        n = len(next(iter(params.values())))
        return torch.zeros(n, NUM_CHANNELS, T)


@pytest.fixture()
def generator(tmp_path):
    val_file = make_val_waveform_file(tmp_path)
    prior_params = {
        "mass_1": torch.ones(N),
        "mass_2": torch.ones(N),
    }

    def mock_prior(n, device=None):
        return {k: v[:n] for k, v in prior_params.items()}

    return _ConcreteGenerator(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        val_waveform_file=val_file,
        training_prior=mock_prior,
    )


def test_sample_returns_correct_shapes(generator):
    X = torch.zeros(N, len(IFOS), 100)  # dummy background, only len() is used
    waveforms, parameters = generator.sample(X)

    assert waveforms.shape == (N, NUM_CHANNELS, T), (
        f"Waveform shape {waveforms.shape} != ({N}, {NUM_CHANNELS}, {T})"
    )
    assert isinstance(parameters, dict), "parameters should be a dict"
    for key, val in parameters.items():
        assert len(val) == N, (
            f"parameters['{key}'] has length {len(val)}, expected {N}"
        )


def test_sample_returns_prior_parameters(generator):
    """sample() must return the exact dict produced by training_prior."""
    X = torch.zeros(N, len(IFOS), 100)
    _, parameters = generator.sample(X)
    assert "mass_1" in parameters
    assert "mass_2" in parameters
    assert torch.equal(parameters["mass_1"], torch.ones(N))
