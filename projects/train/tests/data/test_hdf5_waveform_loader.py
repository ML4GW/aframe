"""
Unit tests for Hdf5WaveformLoader.

Uses tmp_path + h5py to build minimal HDF5 fixtures with a 'parameters' group.
"""

import h5py
import numpy as np
import pytest
import torch

from train.data.waveforms.loader import Hdf5WaveformLoader


def _write_waveform_hdf5(
    path, n: int, T: int, param_values: dict | None = None
):
    """
    Write a minimal waveform HDF5 understood by Hdf5WaveformLoader.

    Layout:
        waveforms/cross  (n, T)
        waveforms/plus   (n, T)
        parameters/{key} (n,)  for each key in param_values
    """
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        wf_grp = f.create_group("waveforms")
        for ch in ("cross", "plus"):
            wf_grp.create_dataset(
                ch, data=rng.standard_normal((n, T)).astype(np.float32)
            )
        pm_grp = f.create_group("parameters")
        if param_values:
            for k, v in param_values.items():
                pm_grp.create_dataset(k, data=np.asarray(v, dtype=np.float32))
        else:
            pm_grp.create_dataset("mass_1", data=np.ones(n, dtype=np.float32))


def _make_loader(fnames, batch_size=8, batches_per_epoch=4, chunk_size=4):
    return Hdf5WaveformLoader(
        fnames=fnames,
        channels=["cross", "plus"],
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        chunk_size=chunk_size,
        path="waveforms",
    )


class TestLoadChunk:
    @pytest.fixture()
    def loader_and_fname(self, tmp_path):
        n, T = 50, 256
        path = tmp_path / "waveforms.hdf5"
        _write_waveform_hdf5(
            path,
            n=n,
            T=T,
            param_values={"mass_1": np.arange(n, dtype=np.float32)},
        )
        loader = _make_loader([path], batch_size=8, chunk_size=8)
        return loader, path

    def test_waveforms_and_params_same_length(self, loader_and_fname):
        loader, fname = loader_and_fname
        wf, params = loader.load_chunk(fname, start=0, size=8)
        n_wf = next(iter(wf.values())).shape[0]
        for k, v in params.items():
            assert v.shape[0] == n_wf, (
                f"params['{k}'].shape[0]={v.shape[0]} != wf length {n_wf}"
            )

    def test_end_of_file_clip(self, loader_and_fname):
        loader, fname = loader_and_fname
        total = loader.sizes[fname]
        wf, params = loader.load_chunk(fname, start=total - 3, size=10)
        expected = 3
        n_wf = next(iter(wf.values())).shape[0]
        assert n_wf == expected, (
            f"Expected {expected} rows after clip, got {n_wf}"
        )
        for v in params.values():
            assert v.shape[0] == expected


class TestSampleBatch:
    @pytest.fixture()
    def loader(self, tmp_path):
        n, T = 200, 128
        path = tmp_path / "waveforms.hdf5"
        _write_waveform_hdf5(path, n=n, T=T)
        # batch_size > chunk_size forces multiple chunks per batch
        return _make_loader([path], batch_size=16, chunk_size=4)

    def test_param_batch_shape(self, loader):
        _, params = loader.sample_batch()
        for k, v in params.items():
            assert v.shape == (16,), f"params['{k}'] shape {v.shape} != (16,)"

    def test_waveform_batch_shape(self, loader):
        waveforms, _ = loader.sample_batch()
        assert waveforms.shape == (16, 2, 128)

    def test_param_dtype_matches_hdf5(self, tmp_path):
        n, T = 100, 64
        path = tmp_path / "typed.hdf5"
        _write_waveform_hdf5(
            path,
            n=n,
            T=T,
            param_values={"mass_1": np.ones(n, dtype=np.float32)},
        )
        loader = _make_loader([path], batch_size=8, chunk_size=4)
        _, params = loader.sample_batch()
        assert params["mass_1"].dtype == torch.float32


def test_multifile_covers_both_files(tmp_path):
    """
    Two files with clearly different param values.  Over many batches,
    observed param values should come from both files.
    """
    n, T = 100, 64
    path_a = tmp_path / "a.hdf5"
    path_b = tmp_path / "b.hdf5"
    _write_waveform_hdf5(
        path_a, n=n, T=T, param_values={"mass_1": np.ones(n, dtype=np.float32)}
    )
    _write_waveform_hdf5(
        path_b,
        n=n,
        T=T,
        param_values={"mass_1": 2 * np.ones(n, dtype=np.float32)},
    )

    loader = _make_loader(
        [path_a, path_b], batch_size=8, batches_per_epoch=40, chunk_size=4
    )

    observed = set()
    for _, params in loader:
        observed.update(params["mass_1"].round().int().tolist())
        if observed == {1, 2}:
            break

    assert 1 in observed and 2 in observed, (
        f"Expected params from both files; got values: {observed}"
    )
