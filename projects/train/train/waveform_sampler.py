import logging
import math
import warnings
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import torch
from ml4gw.distributions import Cosine
from torch.distributions.uniform import Uniform


# TODO: move to ml4gw
class Hdf5WaveformLoader(torch.utils.data.IterableDataset):
    """
    Iterable dataset that loads samples of waveforms
    from a set of HDF5 files.

    It is _strongly_ recommended that these files have been
    written using [chunked storage]
    (https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage).
    This has shown to produce increases in read-time speeds
    of over an order of magnitude.

    Args:
        fnames:
            Paths to HDF5 files from which to sample data.
        channels:
            Datasets to read from the indicated files, which
            will be stacked along dim 1 of the generated batches
            during iteration.
        batch_size:
            Number of samples to load at each iteration.
        batches_per_epoch:
            Number of batches to generate during each call
            to `__iter__`.
        chunk_size:
            Number of samples to load from each file at a time.
            This is useful for reducing I/O overhead when reading.
        path:
            Optional path to location of datasets in hdf5 files.
            `path` should be delimited by forward slashes. If `None`
            it is assumed the datasets are at the root of the file.
    """

    def __init__(
        self,
        fnames: Iterable[Path],
        channels: Iterable[str],
        batch_size: int,
        batches_per_epoch: int,
        chunk_size: int = 1000,
        path: Optional[Path] = None,
    ):
        self.fnames = fnames
        self.channels = channels
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.chunk_size = chunk_size

        if path is not None:
            self.path = path.split("/")
        else:
            self.path = None

        self.sizes = {}
        self.mmap_files = {}
        self.mmap_datasets = {}

        # for each file store the datasets
        # of interest in a dictionary so we
        # can access them at will without needing
        # to reopen the files each time
        for fname in self.fnames:
            f, g = self.open(fname)
            self.mmap_files[fname] = f
            self.mmap_datasets[fname] = {
                channel: g[channel] for channel in self.channels
            }

            # store sizes of each dataset and warn if not chunked;
            # assumes all dsets have same attributes
            # like size and chunking behavior
            dset = self.mmap_datasets[fname][self.channels[0]]
            self.sizes[fname] = len(dset)
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

        self.waveform_size = dset.shape[1]
        self.probs = np.array([i / self.total for i in self.sizes.values()])

    @property
    def num_channels(self):
        return len(self.channels)

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

    def open(self, fname) -> tuple[h5py.File, h5py.Group]:
        f = group = h5py.File(fname, "r")
        if self.path is not None:
            for path in self.path:
                group = group[path]
        return f, group

    def load_chunk(self, fname, start, size):
        end = min(start + size, self.sizes[fname])
        return {
            channel: self.mmap_datasets[fname][channel][start:end]
            for channel in self.channels
        }

    def sample_batch(self):
        # allocate batch up front
        batch = np.zeros(
            (self.batch_size, self.num_channels, self.waveform_size)
        )

        for i in range(self.chunks_per_batch):
            fname = np.random.choice(self.fnames, p=self.probs)

            chunk_size = min(
                self.chunk_size, self.batch_size - i * self.chunk_size
            )

            # select a random starting index for the chunk
            max_start = self.sizes[fname] - chunk_size
            start = np.random.randint(0, max_start + 1)

            # load the chunk and insert it into the batch
            chunk = self.load_chunk(fname, start, chunk_size)
            batch_start = i * self.chunk_size
            batch_end = batch_start + chunk_size

            for i, channel in enumerate(self.channels):
                batch[batch_start:batch_end, i, :] = chunk[channel]

        return torch.tensor(batch)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.sample_batch()


class ChunkedWaveformDataset(torch.utils.data.IterableDataset):
    """
    Wrapper dataset that will loop through chunks of timeseries
    data produced by another iterable and sample subsets
    of waveforms from each chunk.

    Args:
        chunk_it:
            Iterator which will produce batches of waveform
            data to sample subsets from. Should have shape
            `(N, C, T)`, where `N` is the number of waveformns
            to sample from, `C` is the number of channels,
            and `T` is the number of samples along the
            time dimension for each waveform.
        batch_size:
            Number of waveforms to sample at each iteration
        batches_per_chunk:
            Number of batches of waveforms to sample from
            each chunk before moving on to the next one.
    """

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
