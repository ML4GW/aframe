"""
Unit tests for ChunkedWaveformDataset.
"""

import torch

from train.data.waveforms.loader import ChunkedWaveformDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(n: int, T: int, num_channels: int = 2) -> tuple:
    """Return a (waveforms, params) chunk like Hdf5WaveformLoader produces."""
    waveforms = torch.randn(n, num_channels, T)
    params = {"mass_1": torch.arange(n, dtype=torch.float32)}
    return waveforms, params


def _finite_loader(chunks: list[tuple]) -> list:
    """
    Simulate what ChunkedWaveformDataset receives from a DataLoader wrapping
    Hdf5WaveformLoader.  DataLoader with default batch_size=1 collates each
    yielded (waveforms, params) item into a tensor of shape (1, N, C, T) for
    waveforms and (1, N) for each param, so that _next(it) can unpack them as:
        [waveform_chunk], param_dict_chunk = next(it)
    and then waveform_chunk = tensor[0] of shape (N, C, T).
    """
    return [
        (wf.unsqueeze(0), {k: v.unsqueeze(0) for k, v in p.items()})
        for wf, p in chunks
    ]


# ---------------------------------------------------------------------------
# Yielded tuples have consistent batch dimensions
# ---------------------------------------------------------------------------
def test_consistent_batch_dimensions():
    batch_size = 4
    n_chunk = 20
    T = 128
    chunks = _finite_loader([_make_chunk(n_chunk, T)])
    dataset = ChunkedWaveformDataset(
        chunks, batch_size=batch_size, batches_per_chunk=3
    )

    for wf_batch, params_dict in dataset:
        assert wf_batch.shape[0] == batch_size
        for key, val in params_dict.items():
            assert val.shape[0] == batch_size, (
                f"params['{key}'] batch dim {val.shape[0]} != {batch_size}"
            )


# ---------------------------------------------------------------------------
# Params are indexed with the same permutation as waveforms
# ---------------------------------------------------------------------------
def test_params_indexed_same_as_waveforms():
    """
    Waveform identity is encoded in the first time-sample of channel 0
    (wf[i, 0, 0] = i) and also in params['idx'][i] = i.
    After sampling, both must agree on which waveforms were selected.
    """
    batch_size = 5
    n_chunk = 30
    T = 64

    waveforms = torch.zeros(n_chunk, 2, T)
    waveforms[:, 0, 0] = torch.arange(n_chunk, dtype=torch.float32)
    params = {"idx": torch.arange(n_chunk, dtype=torch.float32)}

    chunks = _finite_loader([(waveforms, params)])
    dataset = ChunkedWaveformDataset(
        chunks, batch_size=batch_size, batches_per_chunk=1
    )

    wf_batch, params_dict = next(iter(dataset))

    # The identity stored in waveform[i, 0, 0] must match params['idx'][i]
    wf_identity = wf_batch[:, 0, 0]
    param_identity = params_dict["idx"]
    assert torch.equal(wf_identity, param_identity), (
        "Waveform and param identities differ "
        "— different permutations were used"
    )


# ---------------------------------------------------------------------------
# Dataset exhausts cleanly on StopIteration
# ---------------------------------------------------------------------------
def test_exhausts_cleanly():
    batch_size = 4
    n_chunk = 20
    T = 64
    batches_per_chunk = 3
    num_chunks = 2

    chunks = _finite_loader(
        [_make_chunk(n_chunk, T) for _ in range(num_chunks)]
    )
    dataset = ChunkedWaveformDataset(
        chunks, batch_size=batch_size, batches_per_chunk=batches_per_chunk
    )

    count = 0
    for _ in dataset:
        count += 1

    expected = num_chunks * batches_per_chunk
    assert count == expected, (
        f"Expected {expected} batches from {num_chunks} chunks × "
        f"{batches_per_chunk} batches/chunk; got {count}"
    )
