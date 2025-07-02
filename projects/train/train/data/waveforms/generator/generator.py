from typing import Callable, TYPE_CHECKING

from ..sampler import WaveformSampler
import torch


if TYPE_CHECKING:
    pass


class WaveformGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        training_prior: Callable,
        **kwargs,
    ):
        """
        A torch module for generating waveforms on the fly.
        Args:
            training_prior:
                A callable that returns a prior distribution
                for the parameters of the waveform generator.
        """
        super().__init__(*args, **kwargs)
        self.training_prior, _ = training_prior()

    def get_train_waveforms(self, *_):
        """
        Method is not implemented for this class, as
        waveforms are generated on the fly.
        """
        pass

    def sample(self, X: torch.Tensor):
        N = len(X)
        parameters = self.training_prior.sample(N)
        generation_params = self.convert(parameters)
        generation_params = {
            k: torch.Tensor(v).to(X.device)
            for k, v in generation_params.items()
        }
        hc, hp = self(**generation_params)
        return hc, hp

    def convert(self, parameters: dict) -> dict:
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
