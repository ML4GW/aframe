from typing import TYPE_CHECKING

from ....prior import AframePrior
from ..sampler import WaveformSampler
import torch


if TYPE_CHECKING:
    pass


class WaveformGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        training_prior: AframePrior,
        **kwargs,
    ):
        """
        A torch module for generating waveforms on the fly.
        Args:
            training_prior:
                A callable that takes an integer N and returns a
                dictionary of parameter Tensors, each of length `N`
        """
        super().__init__(*args, **kwargs)
        self.training_prior = training_prior

    def sample(self, X: torch.Tensor):
        N = len(X)
        parameters = self.training_prior(N, device=X.device)
        hc, hp = self(**parameters)
        return hc, hp

    def forward(self):
        raise NotImplementedError
