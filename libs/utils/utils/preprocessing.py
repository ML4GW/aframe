from collections.abc import Callable

import torch
from torch import Tensor

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows


class BackgroundSnapshotter(torch.nn.Module):
    """
    Update a kernel with a new piece of streaming data.

    Maintains a sliding window of data by concatenating new streaming
    updates with the previous snapshot and extracting the appropriate
    sized window for the next iteration. Useful for processing
    continuous data streams in chunks.

    Args:
        psd_length (float): Length of PSD data in seconds.
        kernel_length (float): Length of kernel/window in seconds.
        fduration (float): Duration of whitening filter in seconds.
        sample_rate (float): Sampling rate of data in Hz.
        inference_sampling_rate (float): Sampling rate of network output in Hz.

    Example:
        >>> snapshotter = BackgroundSnapshotter(
        ...     psd_length=64, kernel_length=8, fduration=1,
        ...     sample_rate=2048, inference_sampling_rate=16
        ... )
        >>> update = torch.randn(128, 2, 16384)  # (batch, channels, samples)
        >>> snapshot = torch.randn(128, 2, 262144)
        >>> x, new_snapshot = snapshotter(update, snapshot)
    """

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        # Calculate total state length accounting for PSD, kernel, and filter
        state_length = kernel_length + fduration + psd_length
        # Adjust for inference sampling rate granularity
        state_length -= 1 / inference_sampling_rate
        # Convert to number of samples at the given sample rate
        self.state_size = int(state_length * sample_rate)

    def forward(self, update: Tensor, snapshot: Tensor) -> tuple[Tensor, ...]:
        """
        Concatenate new update with snapshot and extract sliding window.

        Args:
            update (Tensor): New data chunk to append.
            snapshot (Tensor): Previous sliding window state.

        Returns:
            tuple[Tensor, Tensor]:
                - x: Concatenated full data
                - snapshot: Updated sliding window
        """
        # Concatenate new data with previous snapshot
        x = torch.cat([snapshot, update], axis=-1)
        # Extract the sliding window from the end of concatenated data
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot


class PsdEstimator(torch.nn.Module):
    """
    Estimate power spectral density and prepare data for whitening.

    Splits input data into two unequal-length segments: the first segment
    is used to calculate the PSD via spectral density estimation, and the
    second segment is returned for whitening.

    Args:
        length (float): Length of timeseries data in seconds to be returned
            for whitening. The PSD is calculated from the remaining time
            at the beginning of the input.
        sample_rate (float): Sampling rate of input data in Hz.
        fftlength (float): Length of FFTs in seconds when computing the PSD.
        window (Tensor, optional): Window function to apply to FFT segments.
            If None, a default window is used. Defaults to None.
        overlap (float, optional): Overlap between FFT windows in seconds.
            If None, defaults to fftlength / 2. Defaults to None.
        average (str, optional): Method for aggregating spectra from
            FFT windows. Either 'mean' or 'median'. Defaults to 'median'.
        fast (bool, optional): If True, use faster PSD algorithm that is
            inaccurate for the lowest two frequency bins. Safe if highpass
            filtering will be applied later. Defaults to True.

    Example:
        >>> estimator = PsdEstimator(
        ...     length=8.0, sample_rate=2048, fftlength=4.0
        ... )
        >>> X = torch.randn(2, 2, 262144)  # (batch, channels, time)
        >>> X, psds = estimator(X)
    """

    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        window: Tensor | None = None,
        overlap: float | None = None,
        average: str = "median",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        # Initialize spectral density estimator
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, window=window, fast=fast
        )

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
        Split data into PSD estimation and whitening segments.

        Handles the case where input data has two batch elements,
        using the first for PSD calculation and the second for whitening.

        Args:
            X (Tensor): Input data to split

        Returns:
            tuple[Tensor, Tensor]:
                - x: Data segment for whitening
                - psds: Estimated power spectral densities
        """
        # Split data into segments
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

        # Calculate PSDs from background segment
        psds = self.spectral_density(background.double())
        return X, psds


class BatchWhitener(torch.nn.Module):
    """
    Calculate the PSDs and whiten an entire batch of kernels at once.

    Combines PSD estimation and whitening in a single pass.
    Optionally applies augmentation and extracts kernels via unfolding.

    Args:
        kernel_length (float): Length of output kernels in seconds.
        sample_rate (float): Input sampling rate in Hz.
        inference_sampling_rate (float): Sampling rate of network output in Hz.
            Determines the overlap between kernels.
        batch_size (int): Number of kernels to extract from input.
        fduration (float): Duration of the whitening filter in seconds
        fftlength (float): FFT length for PSD calculation in seconds.
        augmentor (Callable, optional): Function to apply augmentation.
            Called with shape (batch, channels, kernel_size). Defaults to None.
        highpass (float, optional): Highpass frequency in Hz. Applied during
            whitening. Defaults to None.
        lowpass (float, optional): Lowpass frequency in Hz. Applied during
            whitening. Defaults to None.
        return_whitened (bool, optional): If True, also return the full
            whitened timeseries before unfolding. Defaults to False.

    Example:
        >>> whitener = BatchWhitener(
        ...     kernel_length=8, sample_rate=2048,
        ...     inference_sampling_rate=16, batch_size=128,
        ...     fduration=2, fftlength=2
        ... )
        >>> x = torch.randn(2, 137216)  # (channels, time)
        >>> kernels = whitener(x)  # shape: (batch_size, channels, kernel_size)
    """

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        augmentor: Callable[[Tensor], Tensor] | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        return_whitened: bool = False,
    ) -> None:
        super().__init__()
        # Calculate stride between kernels based on inference sampling rate
        self.stride_size = int(sample_rate / inference_sampling_rate)
        # Convert kernel length to samples
        self.kernel_size = int(kernel_length * sample_rate)
        self.augmentor = augmentor
        self.return_whitened = return_whitened

        # do length calculations in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate

        # Initialize PSD estimator with calculated total length
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        # Initialize whitening module
        self.whitener = Whiten(fduration, sample_rate, highpass, lowpass)

    def forward(self, x: Tensor) -> Tensor:
        """
        Estimate PSD, whiten data, and unfold kernels.

        Args:
            x (Tensor): Input data of shape (batch, channels, time) or
                (channels, time).

        Returns:
            Tensor: Extracted and optionally augmented kernels of shape
                (batch_size, channels, kernel_size).

            If return_whitened=True, returns tuple of (kernels, whitened_data).

        Raises:
            ValueError: If input is not 2 or 3 dimensional.
        """
        # Determine number of channels for later reshaping
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )

        # Estimate PSD and prepare data
        x, psd = self.psd_estimator(x.double())
        # Apply whitening using estimated PSD
        whitened = self.whitener(x, psd)

        # unfold x and then put it into the expected shape.
        # Note that if x has both signal and background
        # batch elements, they will be interleaved along
        # the batch dimension after unfolding
        x = unfold_windows(whitened, self.kernel_size, self.stride_size)
        # Reshape to (batch_size, channels, kernel_size)
        x = x.reshape(-1, num_channels, self.kernel_size)

        # Apply optional augmentation
        if self.augmentor is not None:
            x = self.augmentor(x)

        if self.return_whitened:
            return x, whitened
        return x


class MultiModalPreprocessor(torch.nn.Module):
    """
    Preprocess data for multimodal model with time and frequency domain inputs.

    Produces both whitened time-domain kernels and frequency-domain kernels
    concatenated with the amplitude spectral density (ASD).

    Args:
        kernel_length (float): Length of output kernels in seconds.
        sample_rate (float): Input sampling rate in Hz.
        inference_sampling_rate (float): Sampling rate of network output in Hz.
            Determines the overlap between kernels.
        batch_size (int): Number of kernels to extract from input.
        fduration (float): Duration of the whitening filter in seconds.
        fftlength (float): FFT length for PSD calculation in seconds.
        highpass (float, optional): Highpass frequency in Hz. Applied during
            whitening and used for frequency masking. Defaults to None.
        lowpass (float, optional): Lowpass frequency in Hz. Applied during
            whitening and used for frequency masking. Defaults to None.

    Example:
        >>> preprocessor = MultiModalPreprocessor(
        ...     kernel_length=8, sample_rate=2048,
        ...     inference_sampling_rate=16, batch_size=128,
        ...     fduration=2, fftlength=2, highpass=32
        ... )
        >>> x = torch.randn(2, 137216)  # (channels, time)
        >>> x_time, x_freq = preprocessor(x)
        >>> # x_time: (batch_size, channels, kernel_size)
        >>> # x_freq: (batch_size, 3*channels, num_freqs)
    """

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        highpass: float | None = None,
        lowpass: float | None = None,
    ) -> None:
        super().__init__()
        # Calculate stride between kernel centers
        self.stride_size = int(sample_rate / inference_sampling_rate)
        # Convert kernel length to samples
        self.kernel_size = int(kernel_length * sample_rate)

        # do length calculations in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate

        # Initialize PSD estimator with calculated total length
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        # Initialize whitening module
        self.whitener = Whiten(fduration, sample_rate, highpass, lowpass)

        # Create frequency mask for filtering frequency bins
        freqs = torch.fft.rfftfreq(self.kernel_size, d=1 / sample_rate)
        self.freq_mask = torch.ones_like(freqs, dtype=torch.bool)
        # Compute highpass frequency mask
        if highpass is not None:
            self.freq_mask &= freqs > highpass
        # Compute lowpass frequency mask
        if lowpass is not None:
            self.freq_mask &= freqs < lowpass

    def forward(self, x: Tensor) -> Tensor:
        """
        Preprocess data for multimodal training.

        Produces both whitened time-domain kernels and frequency-domain kernels
        concatenated with the amplitude spectral density (ASD).

        Args:
            x (Tensor): Input data of shape (batch, channels, time) or
                (channels, time).

        Returns:
            Tuple[Tensor, Tensor]:
                - x_time: Whitened time-domain kernels of shape
                  (batch_size, channels, kernel_size)
                - x_freq: Frequency-domain data of shape
                  (batch_size, 3*channels, num_freqs)
                  where the last 2 channels contain the inverse ASD

        Raises:
            ValueError: If input is not 2 or 3 dimensional.
        """
        # Determine number of channels for later reshaping
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )

        # Estimate PSD and prepare data
        x, psd = self.psd_estimator(x.double())
        # Apply whitening using estimated PSD
        whitened = self.whitener(x, psd)

        # Calculate amplitude spectral density and interpolate
        asd = psd**0.5
        asd = asd.float()
        asd = torch.nn.functional.interpolate(
            asd.unsqueeze(0),
            size=(len(self.freq_mask),),
            mode="linear",
        )
        # Apply frequency mask to filter out unwanted frequency bands
        asd = asd[:, :, self.freq_mask]
        # Scale ASD by 1e23 to bring order of magnitude closer to 1
        asd *= 1e23

        # unfold x and then put it into the expected shape.
        # Note that if x has both signal and background
        # batch elements, they will be interleaved along
        # the batch dimension after unfolding
        x = unfold_windows(whitened, self.kernel_size, self.stride_size)
        # Reshape to (batch_size, channels, kernel_size)
        x = x.reshape(-1, num_channels, self.kernel_size)

        # Expand ASD to match batch size
        asd = asd.expand(x.shape[0], -1, -1)
        inv_asd = 1 / asd

        # Compute FFT of whitened time-domain kernels
        x_fft = torch.fft.rfft(x, dim=-1)
        # Apply frequency mask to FFT
        x_fft = x_fft[:, :, self.freq_mask]
        # Concatenate real part, imaginary part, and inverse ASD
        # Shape: (batch_size, 3*channels, num_freqs)
        x_fft = torch.cat([x_fft.real, x_fft.imag, inv_asd], dim=1)

        return x, x_fft
