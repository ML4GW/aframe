from typing import Callable, Optional, Tuple

import torch
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows

Tensor = torch.Tensor


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


class BatchWhitener(torch.nn.Module):
    """Calculate the PSDs and whiten an entire batch of kernels at once"""

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        augmentor: Optional[Callable] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        return_whitened: bool = False,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.augmentor = augmentor
        self.return_whitened = return_whitened

        # do foreground length calculation in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        self.whitener = Whiten(fduration, sample_rate, highpass, lowpass)

        freqs = torch.fft.rfftfreq(size, d=1 / sample_rate)
        self.freq_mask = (freqs > highpass) & (freqs < lowpass)

    def forward(self, x: Tensor) -> Tensor:
        # Get the number of channels so we know how to
        # reshape `x` appropriately after unfolding to
        # ensure we have (batch, channels, time) shape
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )

        x, psd = self.psd_estimator(x.double())
        whitened = self.whitener(x, psd)

        x = x.float()

        if self.return_asd:
            asd = psd**0.5
            asd = asd.float()
            asd = torch.nn.functional.interpolate(
                asd.unsqueeze(0),
                size=(len(self.freq_mask),),
                mode="linear",
            )
            asd = asd[:, :, self.freq_mask]
            asd = asd.expand(x.shape[0], -1, -1)

        # unfold x and then put it into the expected shape.
        # Note that if x has both signal and background
        # batch elements, they will be interleaved along
        # the batch dimension after unfolding
        x = unfold_windows(whitened, self.kernel_size, self.stride_size)
        x = x.reshape(-1, num_channels, self.kernel_size)
        if self.augmentor is not None:
            x = self.augmentor(x)

        if self.return_whitened and self.return_asd:
            return x, whitened, asd
        elif self.return_whitened:
            return x, whitened
        elif self.return_asd:
            return x, asd
        else:
            return x


class MultiModalPreprocessor(torch.nn.Module):
    """
    Preprocess a batch of waveforms for multimodal training.
    This includes whitening the time domain data and
    calculating the frequency domain data
    """

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)

        # do foreground length calculation in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        self.whitener = Whiten(fduration, sample_rate, highpass, lowpass)

        freqs = torch.fft.rfftfreq(self.kernel_size, d=1 / sample_rate)
        self.freq_mask = torch.ones_like(freqs, dtype=torch.bool)
        if highpass is not None:
            self.freq_mask &= freqs > highpass
        if lowpass is not None:
            self.freq_mask &= freqs < lowpass

    def forward(self, x: Tensor) -> Tensor:
        # Get the number of channels so we know how to
        # reshape `x` appropriately after unfolding to
        # ensure we have (batch, channels, time) shape
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )

        x, psd = self.psd_estimator(x.double())
        whitened = self.whitener(x, psd)

        x = x.float()

        asd = psd**0.5
        asd = asd.float()
        asd = torch.nn.functional.interpolate(
            asd.unsqueeze(0),
            size=(len(self.freq_mask),),
            mode="linear",
        )
        asd = asd[:, :, self.freq_mask]
        asd *= 1e23

        # unfold x and then put it into the expected shape.
        # Note that if x has both signal and background
        # batch elements, they will be interleaved along
        # the batch dimension after unfolding
        x = unfold_windows(whitened, self.kernel_size, self.stride_size)
        x = x.reshape(-1, num_channels, self.kernel_size)

        asd = asd.expand(x.shape[0], -1, -1)
        inv_asd = 1 / asd

        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = x_fft[:, :, self.freq_mask]
        x_fft = torch.cat([x_fft.real, x_fft.imag, inv_asd], dim=1)

        return x, x_fft
