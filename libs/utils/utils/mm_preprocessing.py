from typing import Callable, Optional, Tuple

import torch
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows
import numpy as np
from collections.abc import Sequence

Tensor = torch.Tensor

import torch.nn.functional as F
import torchaudio.transforms as T

class PsdEstimator(torch.nn.Module):
    """
    Module that takes a sample of data, splits it into
    two unequal-length segments, calculates the PSD of
    the first section, then returns this PSD along with
    the second section.

    Args:
        length:
            The length, in seconds, of timeseries data
            to be returned for whitening. Note that the
            length of time used for the PSD will then be
            whatever remains along first part of the time
            axis of the input.
        sample_rate:
            Rate at which input data has been sampled in Hz
        fftlength:
            Length of FFTs to use when computing the PSD
        overlap:
            Amount of overlap between FFT windows when
            computing the PSD. Default value of `None`
            uses `fftlength / 2`
        average:
            Method for aggregating spectra from FFT
            windows, either `"mean"` or `"median"`
        fast:
            If `True`, use a slightly faster PSD algorithm
            that is inaccurate for the lowest two frequency
            bins. If you plan on highpassing later, this
            should be fine.
    """

    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        window: Optional[torch.Tensor] = None,
        overlap: Optional[float] = None,
        average: str = "median",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, window=window, fast=fast
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)

        # if we have 2 batch elements in our input data,
        # it will be assumed that the 0th element is data
        # being used to calculate the psd to whiten the
        # 1st element. Used when we want to use raw background
        # data to calculate the PSDs to whiten data with injected signals
        if X.ndim == 3 and X.size(0) == 2:
            # 0th background element is used to calculate PSDs
            background = background[0]
            # 1st element is the data to be whitened
            X = X[1]

        psds = self.spectral_density(background.double())
        return X, psds

class BackgroundSnapshotter(torch.nn.Module):
    """Update a kernel with a new piece of streaming data"""

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(self, update: Tensor, snapshot: Tensor) -> Tuple[Tensor, ...]:
        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot

class mm_BatchWhitener(torch.nn.Module):
    """Calculate the PSDs and whiten an entire batch of kernels at once"""

    def __init__(
        self,
        resample_rates: Sequence[float], 
        kernel_lengths: Sequence[float], 
        high_passes: Sequence[float], 
        low_passes: Sequence[float],
        inference_sampling_rates: Sequence[float],
        starting_offsets: Sequence[int],
        num_ifos: int,
        kernel_length: float,
        sample_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.resample_rates = resample_rates
        self.stride_sizes = [int(sample_rate / isr) for isr in inference_sampling_rates]
        self.kernel_sizes = [int(kl * sample_rate) for kl in kernel_lengths]
        self.num_timeseries = len(kernel_lengths)-1
        self.num_ifos = num_ifos
        # do foreground length calculation in units of samples,
        # then convert back to length to guard for intification
        self.starting_offsets = []
        self.starting_offsets.append(int(kernel_length*sample_rate-self.kernel_sizes[0]))
        running_offset = 0
        for i, so in enumerate(starting_offsets):
            running_offset += so
            self.starting_offsets.append(int(kernel_length*sample_rate-running_offset*min(self.stride_sizes)-self.kernel_sizes[i+1]))

        self.starting_offsets.append(int(kernel_length*sample_rate-self.kernel_sizes[-1]))
        
        self.ending_offsets = []
        self.ending_offsets.append(None)
        running_offset = 0
        for so in starting_offsets:
            running_offset += so
            if int(running_offset*min(self.stride_sizes)) == 0:
                self.ending_offsets.append(None)
            else:
                self.ending_offsets.append(-int(running_offset*min(self.stride_sizes)))

        self.ending_offsets.append(None)
        
        stride_size = sample_rate / max(inference_sampling_rates)
        self.kernel_size = int(kernel_length * sample_rate)
        strides = (batch_size - 1) * stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=True,
        )
        self.whiteners = torch.nn.ModuleList([Whiten(fduration, sample_rate, highpass, lowpass) 
                                              for highpass, lowpass in zip(high_passes, low_passes)])
        self.resamplers = torch.nn.ModuleList([T.Resample(sample_rate, rr) for rr in resample_rates])
        self.resample_rate = [sample_rate//rr for rr in resample_rates]
        self.fft_highpass = high_passes[-1]
        self.fft_lowpass = low_passes[-1]

    def forward(self, x: Tensor) -> Tensor:
        out_x = tuple()
        x, psd = self.psd_estimator(x)
        for i in range(self.num_timeseries):
            whitened = self.whiteners[i](x.double(), psd)
            sliced_x = whitened[..., self.starting_offsets[i]:self.ending_offsets[i]]
            sliced_x = unfold_windows(sliced_x, self.kernel_sizes[i], self.stride_sizes[i])
            sliced_x = sliced_x.reshape(-1, self.num_ifos, self.kernel_sizes[i])
            bs = sliced_x.shape[0]
            sliced_x = sliced_x.reshape((self.num_ifos*bs, 1, self.kernel_sizes[i])).squeeze(-2)
            sliced_x = self.resamplers[i](sliced_x)
            sliced_x = sliced_x.reshape((bs, self.num_ifos, int(self.kernel_sizes[i]//self.resample_rate[i])))
            out_x = out_x + (sliced_x,)
        
        whitened = self.whiteners[-1](x.double(), psd)
        sliced_x = whitened[..., self.starting_offsets[-1]:self.ending_offsets[-1]]
        sliced_x = unfold_windows(sliced_x, self.kernel_sizes[-1], self.stride_sizes[-1])
        sliced_x = sliced_x.reshape(-1, self.num_ifos, self.kernel_sizes[-1])
        bs = sliced_x.shape[0]
        sliced_x = sliced_x.reshape((self.num_ifos*bs, 1, self.kernel_sizes[-1])).squeeze(-2)
        sliced_x = self.resamplers[-1](sliced_x)
        sliced_x = sliced_x.reshape((bs, self.num_ifos, int(self.kernel_sizes[-1]//self.resample_rate[-1])))
        freqs = torch.fft.rfftfreq(
            sliced_x.shape[-1], d=1 / self.resample_rates[-1]
        )
        sliced_x = torch.fft.rfft(sliced_x)
        mask = freqs >= self.fft_highpass
        mask *= freqs <= self.fft_lowpass
        sliced_x = sliced_x[:, :, mask]
        freqs = np.linspace(0, self.resample_rates[-1]/2, psd.shape[-1])
        mask = freqs >= self.fft_highpass
        mask *= freqs <= self.fft_lowpass
        asds = (psd[..., mask]**0.5 * 1e23).float()
        asds = asds.unsqueeze(dim = 0)
        asds = F.interpolate(asds, size=(sliced_x.shape[-1],), mode="linear", align_corners=False)
        asds = asds.repeat(sliced_x.shape[0], 1, 1)
        sliced_x = torch.cat((sliced_x.real, sliced_x.imag, 1/asds), dim=1)
        out_x = out_x + (sliced_x,)
        return out_x