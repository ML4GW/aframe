import logging
import math
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
        chunk_size: int = 1000,
    ):
        self.fnames = fnames
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.chunk_size = chunk_size

        self.sizes = {}
        self.mmap_files = {}
        self.mmap_datasets = {}

        # determine the number of waveforms up front
        for fname in self.fnames:
            f = h5py.File(fname, "r")
            self.mmap_files[fname] = f

            dset_cross = f["waveforms"]["cross"]
            dset_plus = f["waveforms"]["plus"]

            self.mmap_datasets[fname] = {
                "cross": dset_cross,
                "plus": dset_plus,
            }
            self.sizes[fname] = len(dset_cross)

            if dset_cross.chunks is None:
                warnings.warn(
                    "File {} contains datasets that were generated "
                    "without using chunked storage. This can have "
                    "severe performance impacts at data loading time. "
                    "If you need faster loading, try re-generating "
                    "your datset with chunked storage turned on.".format(
                        fnames
                    ),
                )

        self.waveform_size = dset_cross.shape[1]
        self.num_pols = 2
        self.probs = np.array([i / self.total for i in self.sizes.values()])

    @property
    def chunks_per_batch(self):
        return math.ceil(self.batch_size / self.chunk_size)

    @property
    def total(self):
        return sum(self.sizes.values())

    def __len__(self):
        return self.batches_per_epoch

    def __del__(self):
        # close all opened files when the object is destroyed
        for f in self.mmap_files.values():
            f.close()

    def sample_fnames(self) -> np.ndarray:
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=(self.chunks_per_batch,),
            replace=True,
        )

    def load_chunk(self, fname, start, size):
        end = min(start + size, self.sizes[fname])
        return {
            "cross": self.mmap_datasets[fname]["cross"][start:end],
            "plus": self.mmap_datasets[fname]["plus"][start:end],
        }

    def sample_batch(self):
        # allocate batch up front
        batch = np.zeros((self.batch_size, self.num_pols, self.waveform_size))

        for i in range(self.chunks_per_batch):
            fname = np.random.choice(self.fnames, p=self.probs)

            chunk_size = min(
                self.chunk_size, self.batch_size - i * self.chunk_size
            )

            # select a random starting index for the chunk
            max_start = self.sizes[fname] - chunk_size
            start = np.random.randint(0, max_start + 1)

            # Load the chunk
            chunk = self.load_chunk(fname, start, chunk_size)
            # Add the chunk to the batch
            batch_start = i * self.chunk_size
            batch_end = batch_start + chunk_size
            batch[batch_start:batch_end, 0, :] = chunk["cross"]
            batch[batch_start:batch_end, 1, :] = chunk["plus"]

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
        self.logger = logging.getLogger(__name__)
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
                idx = torch.randperm(num_waveforms)[: self.batch_size]
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
