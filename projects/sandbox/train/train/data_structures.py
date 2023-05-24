from typing import Callable, Optional, Tuple

import numpy as np
import torch

from ml4gw import gw
from ml4gw.dataloading import InMemoryDataset
from ml4gw.distributions import PowerLaw
from ml4gw.spectral import Background, normalize_psd
from ml4gw.transforms.transform import FittableTransform
from ml4gw.utils.slicing import sample_kernels


class ChannelSwapper(torch.nn.Module):
    """
    Data augmentation module that randomly swaps channels
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels swapped.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            num = num if not num % 2 else num - 1
            num = max(2, num)
            channel = torch.randint(X.shape[1], size=(num // 2,)).repeat(2)
            # swap channels from the first num / 2 elements with the
            # second num / 2 elements
            print(indices, X.shape)
            indices = torch.arange(num)
            target_indices = torch.roll(indices, shifts=num // 2, dims=0)
            X[indices, channel] = X[target_indices, channel]

        return X, indices


class ChannelMuter(torch.nn.Module):
    """
    Data augmentation module that randomly mutes 1 channel
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels muted.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            channel = torch.randint(X.shape[1], size=(num,))
            indices = torch.randint(X.shape[0], size=(num,))
            X[indices, channel] = torch.zeros(X.shape[-1], device=X.device)

        return X, indices


class AframeInMemoryDataset(InMemoryDataset):
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
        glitch_channels = len(list(self.buffers()))
        if X.shape[1] < glitch_channels:
            raise ValueError(
                "Can't insert glitches into tensor with {} channels "
                "using glitches from {} ifos".format(
                    X.shape[1], glitch_channels
                )
            )

        # sample batch indices which will be replaced with
        # a glitch independently from each interferometer
        masks = torch.rand(size=(glitch_channels, len(X))) < self.prob
        for i, glitches in enumerate(self.buffers()):
            mask = masks[i]

            # now sample from our bank of glitches for this
            # interferometer the number we want to insert
            N = mask.sum().item()
            idx = torch.randint(len(glitches), size=(N,))

            # finally sample kernels from the selected glitches.
            # Add a dummy dimension so that sample_kernels
            # doesn't think this is a single multi-channel
            # timeseries, but rather a batch of single
            # channel timeseries
            glitches = glitches[idx, None]
            glitches = sample_kernels(
                glitches,
                kernel_size=X.shape[-1],
                max_center_offset=self.max_offset,
            )

            # replace the appropriate channel in our
            # strain data with the sampled glitches
            X[mask, i] = glitches[:, 0]

            # use bash file permissions style
            # numbers to indicate which channels
            # go inserted on
            y[mask] -= 2 ** (i + 1)
        return X, y


class SignalInverter(torch.nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        if self.training:
            mask = torch.rand(size=X.shape[:-1]) < self.prob
            X[mask] *= -1
        return X


class SignalReverser(torch.nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        if self.training:
            mask = torch.rand(size=X.shape[:-1]) < self.prob
            X[mask] = X[mask].flip(-1)
        return X


# TODO: use ml4gw version if/when it's merged
class SnrRescaler(FittableTransform):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        waveform_duration: float,
        highpass: Optional[float] = None,
    ):
        super().__init__()
        self.highpass = highpass
        self.sample_rate = sample_rate
        self.df = 1 / waveform_duration
        waveform_size = int(waveform_duration * sample_rate)
        num_freqs = int(waveform_size // 2 + 1)
        buff = torch.zeros((num_ifos, num_freqs), dtype=torch.float64)
        self.register_buffer("background", buff)

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer("mask", freqs >= highpass, persistent=False)
        else:
            self.mask = None

    def fit(
        self,
        *backgrounds: Background,
        sample_rate: Optional[float] = None,
        fftlength: float = 2
    ):
        psds = []
        for background in backgrounds:
            psd = normalize_psd(
                background, self.df, self.sample_rate, sample_rate, fftlength
            )
            psds.append(psd)

        background = torch.tensor(np.stack(psds), dtype=torch.float64)
        super().build(background=background)

    def forward(
        self,
        responses: gw.WaveformTensor,
        target_snrs: Optional[gw.ScalarTensor] = None,
    ):
        snrs = gw.compute_network_snr(
            responses, self.background, self.sample_rate, self.mask
        )
        if target_snrs is None:
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]

        target_snrs.to(snrs.device)
        weights = target_snrs / snrs
        rescaled_responses = responses * weights.view(-1, 1, 1)

        return rescaled_responses, target_snrs


class SnrSampler:
    def __init__(
        self,
        max_min_snr: float,
        min_min_snr: float,
        max_snr: float,
        alpha: float,
        decay_steps: int,
    ):
        self.max_min_snr = max_min_snr
        self.min_min_snr = min_min_snr
        self.max_snr = max_snr
        self.alpha = alpha
        self.decay_steps = decay_steps
        self._step = 0

        self.dist = PowerLaw(max_min_snr, max_snr, alpha)

    def __call__(self, N):
        return self.dist(N)

    def step(self):
        self._step += 1
        if self._step >= self.decay_steps:
            return

        frac = self._step / self.decay_steps
        diff = self.max_min_snr - self.min_min_snr
        new = self.max_min_snr - frac * diff

        self.dist.x_min = new
        self.dist.normalization = new ** (-self.alpha + 1)
        self.dist.normalization -= self.max_snr ** (-self.alpha + 1)

        self._step += 1
