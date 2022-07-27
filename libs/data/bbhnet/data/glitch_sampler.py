from typing import Optional

import h5py
import numpy as np
import torch

from bbhnet.data.utils import sample_kernels


# TODO: generalize to arbitrary ifos
class GlitchSampler:
    def __init__(
        self,
        glitch_dataset: str,
        deterministic: bool = False,
        frac: Optional[float] = None,
    ) -> None:
        # TODO: will these need to be resampled?
        with h5py.File(glitch_dataset, "r") as f:
            hanford_glitches = f["H1_glitches"][:]
            livingston_glitches = f["L1_glitches"][:]

        if frac is not None:
            num_hanford_glitches = int(frac * len(hanford_glitches))
            num_livingston_glitches = int(frac * len(livingston_glitches))
            if frac < 0:
                hanford_glitches = hanford_glitches[num_hanford_glitches:]
                livingston_glitches = livingston_glitches[
                    num_livingston_glitches:
                ]
            else:
                hanford_glitches = hanford_glitches[:num_hanford_glitches]
                livingston_glitches = livingston_glitches[
                    :num_livingston_glitches
                ]

        self.hanford = torch.Tensor(hanford_glitches)
        self.livingston = torch.Tensor(livingston_glitches)

        self.deterministic = deterministic

    def to(self, device: str) -> None:
        self.hanford = self.hanford.to(device)
        self.livingston = self.livingston.to(device)

    def sample(self, N: int, size: int, offset: int = 0) -> np.ndarray:
        """Sample glitches from each interferometer

        If `self.deterministic` is `True`, this will grab the first
        `N` glitches from each interferometer, with the center of
        each kernel placed at the trigger time minus some specified
        amount of offset.

        If `self.deterministic` is `False`, this will sample _at most_
        `N` kernels from each interferometer, with the _total_ glitches
        sampled equal to `N`. The sampled glitches will be chosen at
        random, and the kernel sampled from each glitch will be randomly
        selected, with `offset` indicating the maximum distance the right
        edge of the kernel can be from the trigger time, i.e. the default
        value of 0 indicates that every kernel must contain the trigger.
        """

        if self.deterministic:
            if N == -1:
                N = len(self.hanford)

            center = int(self.hanford.shape[-1] // 2)
            left = int(center + offset - size // 2)
            right = int(left + size)

            hanford = self.hanford[:N, left:right]
            livingston = self.livingston[:N, left:right]
        else:
            hanford = livingston = None
            num_hanford = np.random.randint(N)
            num_livingston = N - num_hanford

            if num_hanford > 0:
                hanford = sample_kernels(
                    self.hanford, size, offset, num_hanford
                )
                hanford = torch.stack(hanford, axis=0)

            if num_livingston > 0:
                livingston = sample_kernels(
                    self.livingston, size, offset, num_livingston
                )
                livingston = torch.stack(livingston, axis=0)

        return hanford, livingston
