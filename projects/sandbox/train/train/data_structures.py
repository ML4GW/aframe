from typing import Callable, Optional, Tuple

import numpy as np
import torch

from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms.injection import RandomWaveformInjection
from ml4gw.utils.slicing import sample_kernels


class BBHInMemoryDataset(InMemoryDataset):
    """
    Dataloader which samples batches of kernels
    from a single timeseries array and prepares
    corresponding target array of all 0s. Optionally
    applies a preprocessing step to both the sampled
    kernels and their targets.

    Args:
        X: Array containing multi-channel timeseries data
        kernel_size:
            The size of the kernels, in terms of number of
            samples, to sample from the timeseries.
        batch_size:
            Number of kernels to produce at each iteration.
            Represents the 0th dimension of the returned tensor.
        batches_per_epoch:
            Number of iterations dataset will perform before
            raising a `StopIteration` exception.
        preprocessor:
            Optional preprocessing step to apply to both the
            sampled kernels and their targets. If left as
            `None`, the batches and targets will be returned
            as-is.
        coincident:
            Whether to sample kernels from all channels using
            the same timesteps, or whether to sample them
            independently from across the whole timeseries.
        shuffle:
            Whether to samples kernels uniformly from the
            timeseries, or iterate through them in order.
        device:
            Device on which to host the timeseries dataset.
    """

    def __init__(
        self,
        X: np.ndarray,
        kernel_size: int,
        batch_size: int = 32,
        batches_per_epoch: Optional[int] = None,
        preprocessor: Optional[Callable] = None,
        coincident: bool = True,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            X,
            kernel_size,
            batch_size=batch_size,
            stride=1,
            batches_per_epoch=batches_per_epoch,
            coincident=coincident,
            shuffle=shuffle,
            device=device,
        )
        self.preprocessor = preprocessor

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = super().__next__()
        y = torch.zeros((len(X), 1)).to(X.device)

        if self.preprocessor is not None:
            X, y = self.preprocessor(X, y)
        return X, y


class BBHNetWaveformInjection(RandomWaveformInjection):
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return X, y

        X, idx, _ = super().forward(X)
        y[idx] = 1
        return X, y


class GlitchSampler(torch.nn.Module):
    def __init__(
        self, prob: float, max_offset: int, **glitches: np.ndarray
    ) -> None:
        super().__init__()
        for ifo, glitch in glitches.items():
            self.register_buffer(ifo, torch.Tensor(glitch))

        self.prob = prob
        self.max_offset = max_offset

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if X.shape[1] < len(self.glitches):
            raise ValueError(
                "Can't insert glitches into tensor with {} channels "
                "using glitches from {} ifos".format(
                    X.shape[1], len(self.glitches)
                )
            )

        # sample batch indices which will be replaced with
        # a glitch independently from each interferometer
        masks = torch.rand(size=(len(self.glitches), len(X))) < self.prob
        for i, ifo in enumerate(self.glitches):
            mask = masks[i]

            # now sample from our bank of glitches for this
            # interferometer the number we want to insert
            N = mask.sum().item()
            idx = torch.randint(len(ifo), size=(N,))

            # finally sample kernels from the selected glitches.
            # Add a dummy dimension so that sample_kernels
            # doesn't think this is a single multi-channel
            # timeseries, but rather a batch of single
            # channel timeseries
            glitches = ifo[idx, None]
            glitches = sample_kernels(
                glitches,
                kernel_size=X.shape[-1],
                max_center_offset=self.max_offset,
            )

            # replace the appropriate channel in our
            # strain data with the sampled glitches
            X[mask, i] = glitches[:, 0]
        return X, y
