import warnings
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.distributions.uniform import Uniform

from ml4gw.distributions import Cosine


class Hdf5WaveformLoader(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fnames: Iterable[Path],
        batch_size: int,
        batches_per_epoch: int,
    ):
        self.fnames = fnames
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        self.sizes = {}
        # determine the number of waveforms up front
        for fname in self.fnames:
            with h5py.File(fname, "r") as f:
                dset = f["waveforms"]["cross"]
                if dset.chunks is None:
                    warnings.warn(
                        "File {} contains datasets that were generated "
                        "without using chunked storage. This can have "
                        "severe performance impacts at data loading time. "
                        "If you need faster loading, try re-generating "
                        "your datset with chunked storage turned on.".format(
                            fnames
                        ),
                    )
                self.sizes[fname] = len(dset)

        self.waveform_size = dset.shape[1]
        self.num_pols = 2
        self.probs = np.array([i / self.total for i in self.sizes.values()])

    def __len__(self):
        return self.batches_per_epoch

    @property
    def total(self):
        return sum(self.sizes.values())

    def sample_fnames(self) -> np.ndarray:
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=(self.batch_size,),
            replace=True,
        )

    def sample_batch(self):
        fnames = self.sample_fnames()
        # allocate batch up front
        batch = np.empty((self.batch_size, self.num_pols, self.waveform_size))

        unique_fnames, inv, counts = np.unique(
            fnames, return_inverse=True, return_counts=True
        )
        for i, (fname, _) in enumerate(zip(unique_fnames, counts)):
            size = self.sizes[fname]

            batch_idx = np.where(inv == i)[0]

            idx = np.random.randint(size, size=len(batch_idx))

            with h5py.File(fname, "r") as f:
                g = f["waveforms"]
                for b, i in zip(batch_idx, idx):
                    for j, pol in enumerate(["cross", "plus"]):
                        batch[b, j] = g[pol][i]
            return torch.tensor(batch)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.sample_batch()


class ChunkedWaveformDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        chunk_it: Iterable,
        batch_size: int,
        batches_per_chunk: int,
    ) -> None:
        self.chunk_it = chunk_it
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk

    def __len__(self):
        return len(self.chunk_it) * self.batches_per_chunk

    def __iter__(self):
        it = iter(self.chunk_it)
        [chunk] = next(it)

        num_waveforms, _, _ = chunk.shape
        while True:
            # generate batches from the current chunk
            for _ in range(self.batches_per_chunk):
                idx = torch.randperm(num_waveforms)[:num_waveforms]
                yield chunk[idx]

            try:
                [chunk] = next(it)
            except StopIteration:
                break
            num_waveforms, _, _ = chunk.shape


class WaveformSampler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dec = Cosine()
        self.psi = Uniform(0, torch.pi)
        self.phi = Uniform(-torch.pi, torch.pi)

    def forward(self, X, prob, waveforms):
        # determine batch size from X and prob
        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < prob
        N = mask.sum().item()

        # sample sky parameters for each injections
        dec = self.dec.sample((N,)).to(X.device)
        psi = self.psi.sample((N,)).to(X.device)
        phi = self.phi.sample((N,)).to(X.device)

        # now sample the actual waveforms we want to inject
        idx = torch.randperm(waveforms.shape[0])[:N]
        waveforms = waveforms[idx].to(X.device).float()

        cross, plus = waveforms[:, 0], waveforms[:, 1]
        polarizations = {"cross": cross, "plus": plus}

        return dec, psi, phi, polarizations, mask
