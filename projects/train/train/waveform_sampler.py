import logging
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
        self, fnames: Iterable[Path], waveforms_per_chunk: int, num_chunks: int
    ):
        self.fnames = fnames
        self.waveforms_per_chunk = waveforms_per_chunk
        self.num_chunks = num_chunks

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
        return self.num_chunks

    @property
    def total(self):
        return sum(self.sizes.values())

    def sample_fnames(self) -> np.ndarray:
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=(self.waveforms_per_chunk,),
            replace=True,
        )

    def sample_batch(self):
        fnames = self.sample_fnames()
        # allocate batch up front
        batch = np.empty(
            (self.waveforms_per_chunk, self.num_pols, self.waveform_size)
        )

        unique_fnames, inv, counts = np.unique(
            fnames, return_inverse=True, return_counts=True
        )
        for i, (fname, count) in enumerate(zip(unique_fnames, counts)):
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
        for _ in range(self.num_chunks):
            yield self.sample_batch()


class WaveformSampler(torch.nn.Module):
    def __init__(
        self,
        fname: Path,
        waveforms_per_chunk: int,
        batches_per_chunk: int,
        batches_per_epoch: int,
    ) -> None:
        super().__init__()
        self.dec = Cosine()
        self.psi = Uniform(0, torch.pi)
        self.phi = Uniform(-torch.pi, torch.pi)

        num_chunks = batches_per_epoch // batches_per_chunk
        chunk_dataset = Hdf5WaveformLoader(
            fname, waveforms_per_chunk, num_chunks=num_chunks
        )
        logging.info(f"Training with {chunk_dataset.total} waveforms")
        self.chunk_loader = iter(
            torch.utils.data.DataLoader(
                chunk_dataset, num_workers=3, batch_size=None
            )
        )

        self.chunk = next(self.chunk_loader)

        self.waveforms_per_chunk = waveforms_per_chunk
        self.batches_per_chunk = batches_per_chunk
        self.batch_num = 0

    def __len__(self):
        return len(self.chunk_it) * self.batches_per_chunk

    def forward(self, X, prob):
        # determine batch size from X and prob
        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < prob
        N = mask.sum().item()

        # sample sky parameters for each injections
        dec = self.dec.sample((N,)).to(X.device)
        psi = self.psi.sample((N,)).to(X.device)
        phi = self.phi.sample((N,)).to(X.device)

        # now sample the actual waveforms we want to inject
        idx = torch.randperm(self.waveforms_per_chunk)[:N]
        waveforms = self.chunk[idx].to(X.device).float()

        cross, plus = waveforms[:, 0], waveforms[:, 1]
        polarizations = {"cross": cross, "plus": plus}
        self.batch_num += 1

        if self.batch_num > self.batches_per_chunk:
            # move onto next chunk
            self.chunk = next(self.chunk_loader)
            self.batch_num = 0

        return dec, psi, phi, polarizations, mask
