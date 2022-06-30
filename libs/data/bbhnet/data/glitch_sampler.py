import h5py
import numpy as np
import torch

from bbhnet.data.utils import sample_kernels


# TODO: generalize to arbitrary ifos
class GlitchSampler:
    def __init__(self, glitch_dataset: str) -> None:
        # TODO: will these need to be resampled?
        with h5py.File(glitch_dataset, "r") as f:
            hanford_glitches = f["H1_glitches"][:]
            livingston_glitches = f["L1_glitches"][:]

        self.hanford = torch.Tensor(hanford_glitches)
        self.livingston = torch.Tensor(livingston_glitches)

    def to(self, device: str) -> None:
        self.hanford = self.hanford.to(device)
        self.livingston = self.livingston.to(device)

    def sample(
        self, N: int, size: int, trigger_distance_size: int = 0
    ) -> np.ndarray:
        num_hanford = np.random.randint(N)
        num_livingston = N - num_hanford

        if num_hanford > 0:
            hanford = sample_kernels(
                self.hanford, size, trigger_distance_size, num_hanford
            )
            hanford = torch.stack(hanford, axis=0)
        else:
            hanford = None

        if num_livingston > 0:
            livingston = sample_kernels(
                self.livingston, size, trigger_distance_size, num_livingston
            )
            livingston = torch.stack(livingston, axis=0)
        else:
            livingston = None
        return hanford, livingston
