"""
Unit tests for WaveformSampler.get_val_waveforms.

Uses minimal mock WaveformSet-like objects and real HDF5 fixtures to
avoid touching the filesystem beyond what's needed.
"""

import math
from dataclasses import fields

from ledger.injections import WaveformSet, waveform_class_factory

from train.data.waveforms.sampler import WaveformSampler

from conftest import (
    IFOS,
    SAMPLE_RATE,
    make_val_waveform_file,
)


def _make_sampler(val_waveform_file):
    """Instantiate a minimal concrete WaveformSampler subclass."""

    class _Sampler(WaveformSampler):
        def sample(self, X):
            raise NotImplementedError

        def get_test_waveforms(self):
            raise NotImplementedError

    return _Sampler(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        val_waveform_file=val_waveform_file,
    )


def test_only_parameter_fields_returned(tmp_path):
    """
    The val waveform file contains parameter, waveform, and metadata fields.
    Only fields with kind='parameter' should appear in the returned params
    dict.
    """
    val_file = make_val_waveform_file(tmp_path)
    sampler = _make_sampler(val_file)

    WS = waveform_class_factory(IFOS, WaveformSet, "WaveformSet")
    _, params = sampler.get_val_waveforms(world_size=1, rank=0)

    parameter_names = {
        f.name for f in fields(WS) if f.metadata.get("kind") == "parameter"
    }
    waveform_names = {
        f.name for f in fields(WS) if f.metadata.get("kind") == "waveform"
    }

    for key in params:
        assert key in parameter_names, (
            f"Returned key '{key}' is not a 'parameter' field"
        )

    assert not waveform_names.intersection(params.keys()), (
        "Waveform-kind fields leaked into params dict"
    )


def test_non_ndarray_params_skipped(tmp_path):
    """
    If a parameter field holds a scalar instead of an ndarray,
    get_val_waveforms should skip it silently (no KeyError / TypeError).
    """
    from unittest.mock import MagicMock, PropertyMock, patch

    val_file = make_val_waveform_file(tmp_path)
    sampler = _make_sampler(val_file)

    WS = waveform_class_factory(IFOS, WaveformSet, "WaveformSet")
    real_ws = WS.read(val_file)

    target_field = next(
        f.name
        for f in fields(real_ws)
        if f.metadata.get("kind") == "parameter"
    )
    setattr(real_ws, target_field, 42.0)

    mock_cls = MagicMock()
    mock_cls.read.return_value = real_ws

    with patch.object(
        type(sampler),
        "waveform_set_cls",
        new_callable=PropertyMock,
        return_value=mock_cls,
    ):
        _, params = sampler.get_val_waveforms(world_size=1, rank=0)

    assert target_field not in params, (
        f"Scalar field '{target_field}' should have been skipped"
    )


def test_world_size_slicing_consistent(tmp_path):
    """
    With world_size=2, rank=0: waveforms and every param tensor should
    have the same length (half of total, rounded up).
    """
    n_total = 20
    val_file = make_val_waveform_file(tmp_path, n=n_total)
    sampler = _make_sampler(val_file)

    waveforms, params = sampler.get_val_waveforms(world_size=2, rank=0)

    expected_len = math.ceil(n_total / 2)
    assert len(waveforms) == expected_len, (
        f"Expected {expected_len} waveforms for rank 0, got {len(waveforms)}"
    )
    for key, val in params.items():
        assert len(val) == len(waveforms), (
            f"params['{key}'] length {len(val)} != "
            f"waveforms length {len(waveforms)}"
        )
