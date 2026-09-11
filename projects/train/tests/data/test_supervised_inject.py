"""
Unit tests for SupervisedAframeDataset.inject.

Each test instantiates a TimeDomainSupervisedAframeDataset (which calls
super().inject via SupervisedAframeDataset) with fixtures from conftest,
calls build_transforms() to wire up the real PSD estimator / whitener /
projector, then calls inject() directly.
"""

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


def _make_dataset(
    data_dir,
    val_waveform_file,
    training_waveform_file,
    waveform_prob=1.0,
    swap_prob=None,
    mute_prob=None,
):
    sampler = WaveformLoader(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        val_waveform_file=val_waveform_file,
        training_waveform_path=training_waveform_file,
    )
    kwargs = base_dataset_kwargs(data_dir, sampler)
    kwargs["waveform_prob"] = waveform_prob
    kwargs["swap_prob"] = swap_prob
    kwargs["mute_prob"] = mute_prob
    dataset = TimeDomainSupervisedAframeDataset(**kwargs)
    setup_dataset_manually(dataset)
    return dataset


class TestNanPlacement:
    def test_all_injected_no_nan(
        self, data_dir, val_waveform_file, training_waveform_file
    ):
        """waveform_prob=1 and no augmentations → no NaN in params."""
        ds = _make_dataset(
            data_dir,
            val_waveform_file,
            training_waveform_file,
            waveform_prob=1.0,
            swap_prob=None,
            mute_prob=None,
        )
        X = make_synthetic_X()
        waveforms = make_synthetic_waveforms(n=32, sliced=True)
        params = {"mass_1": torch.ones(32)}

        _, y, params_out = ds.inject(X=X, waveforms=waveforms, params=params)
        for key, val in params_out.items():
            assert not torch.isnan(val).any(), (
                f"Unexpected NaN in '{key}' when all samples"
                " should be injected"
            )

    def test_fully_swapped_all_nan(
        self, data_dir, val_waveform_file, training_waveform_file
    ):
        """
        waveform_prob=1, swap_prob=1 → all injected samples are swapped →
        mask is reset to all-False → all params must be NaN.
        """
        ds = _make_dataset(
            data_dir,
            val_waveform_file,
            training_waveform_file,
            waveform_prob=1.0,
            swap_prob=1.0,
            mute_prob=None,
        )
        X = make_synthetic_X()
        waveforms = make_synthetic_waveforms(n=32, sliced=True)
        params = {"mass_1": torch.ones(32)}

        _, y, params_out = ds.inject(X=X, waveforms=waveforms, params=params)
        assert y.sum() == 0, "Expected all labels 0 after swap_prob=1"
        for key, val in params_out.items():
            assert torch.isnan(val).all(), (
                f"Expected all NaN in '{key}' after swap_prob=1"
            )

    def test_nan_matches_label_zero(
        self, data_dir, val_waveform_file, training_waveform_file
    ):
        """NaN mask equals label==0 for an arbitrary waveform_prob."""
        ds = _make_dataset(
            data_dir,
            val_waveform_file,
            training_waveform_file,
            waveform_prob=1.0,
            swap_prob=0.25,
            mute_prob=None,
        )
        X = make_synthetic_X()
        waveforms = make_synthetic_waveforms(n=32, sliced=True)
        params = {"mass_1": torch.ones(32)}

        _, y, params_out = ds.inject(X=X, waveforms=waveforms, params=params)
        not_injected = y.squeeze() == 0
        for key, val in params_out.items():
            assert torch.equal(torch.isnan(val), not_injected), (
                f"NaN mask for '{key}' does not match label==0"
            )


def test_extrinsic_keys_always_present(
    data_dir, val_waveform_file, training_waveform_file
):
    """Even with an empty input params dict the four extrinsic keys appear."""
    ds = _make_dataset(
        data_dir, val_waveform_file, training_waveform_file, waveform_prob=1.0
    )
    X = make_synthetic_X()
    waveforms = make_synthetic_waveforms(n=32, sliced=True)

    _, _, params_out = ds.inject(X=X, waveforms=waveforms, params={})

    for key in ("dec", "psi", "phi", "snr"):
        assert key in params_out, (
            f"Expected extrinsic key '{key}' in params_out"
        )
        assert params_out[key].shape == (BATCH_SIZE,)


def test_param_slicing_consistent_with_waveforms(
    data_dir, val_waveform_file, training_waveform_file
):
    """
    Encode the waveform index in a scalar param (params['wf_idx'][i] = i).
    After inject(), injected positions must contain values drawn from
    {0 … N-1} with no repeats, confirming that params and waveforms
    were indexed by the same randperm.
    """
    ds = _make_dataset(
        data_dir,
        val_waveform_file,
        training_waveform_file,
        waveform_prob=1.0,
        swap_prob=None,
        mute_prob=None,
    )
    N = 32
    X = make_synthetic_X()
    waveforms = make_synthetic_waveforms(n=N, sliced=True)
    params = {"wf_idx": torch.arange(N, dtype=torch.float32)}

    _, y, params_out = ds.inject(X=X, waveforms=waveforms, params=params)

    injected_mask = y.squeeze() == 1
    selected_vals = params_out["wf_idx"][injected_mask]

    assert (selected_vals >= 0).all() and (selected_vals < N).all()
    assert selected_vals.numel() == selected_vals.unique().numel()


def test_swap_makes_params_nan(
    data_dir, val_waveform_file, training_waveform_file
):
    """
    With swap_prob=1.0 every injected waveform gets swapped, so the
    final mask is all-False and every param entry must be NaN.
    """
    ds = _make_dataset(
        data_dir,
        val_waveform_file,
        training_waveform_file,
        waveform_prob=1.0,
        swap_prob=1.0,
        mute_prob=None,
    )
    X = make_synthetic_X()
    waveforms = make_synthetic_waveforms(n=32, sliced=True)
    params = {"mass_1": torch.ones(32)}

    _, y, params_out = ds.inject(X=X, waveforms=waveforms, params=params)

    assert y.sum() == 0, "Expected all labels to be 0 after full swap"
    for key, val in params_out.items():
        assert torch.isnan(val).all(), (
            f"Expected all NaN in '{key}' after swap_prob=1.0"
        )
