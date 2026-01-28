from utils.preprocessing import (
    BackgroundSnapshotter,
    PsdEstimator,
    BatchWhitener,
    MultiModalPreprocessor,
    TimeSpectrogramPreprocessor,
)
import torch
import pytest


class TestBackgroundSnapshotter:
    """Test suite for BackgroundSnapshotter module."""

    psd_length = 64
    kernel_length = 8
    fduration = 1
    sample_rate = 2048
    inference_sampling_rate = 16

    def test_initialization(self):
        snapshotter = BackgroundSnapshotter(
            psd_length=self.psd_length,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
        )
        assert snapshotter.state_size == int(
            (
                self.psd_length
                + self.kernel_length
                + self.fduration
                - 1 / self.inference_sampling_rate
            )
            * self.sample_rate
        )

    def test_forward_shape(self):
        snapshotter = BackgroundSnapshotter(
            psd_length=self.psd_length,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
        )
        batch_size, channels, update_samples = 5, 2, 2048
        snapshot_size = snapshotter.state_size

        update = torch.randn(batch_size, channels, update_samples)
        snapshot = torch.randn(batch_size, channels, snapshot_size)

        x, new_snapshot = snapshotter(update, snapshot)

        # Check concatenated output has correct total length
        expected_x_shape = (
            batch_size,
            channels,
            snapshot_size + update_samples,
        )
        assert x.shape == expected_x_shape
        assert new_snapshot.shape == (batch_size, channels, snapshot_size)

    def test_snapshot_output_values(self):
        snapshotter = BackgroundSnapshotter(
            psd_length=self.psd_length,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
        )
        batch_size, channels = 10, 2

        # First iteration
        update = torch.ones(batch_size, channels, snapshotter.state_size)
        snapshot = torch.zeros(batch_size, channels, 2048)
        x, new_snap = snapshotter(update, snapshot)

        assert torch.allclose(new_snap, update)


class TestPsdEstimator:
    """Test suite for PsdEstimator module."""

    length = 8
    sample_rate = 2048
    fftlength = 2

    def test_initialization(self):
        estimator = PsdEstimator(
            length=self.length,
            sample_rate=self.sample_rate,
            fftlength=self.fftlength,
        )
        assert estimator.size == self.length * self.sample_rate

    def test_forward_shape(self):
        psd_length = 64
        estimator = PsdEstimator(
            length=self.length,
            sample_rate=self.sample_rate,
            fftlength=self.fftlength,
        )

        x = torch.randn(2, (psd_length + self.length) * self.sample_rate)
        x, psds = estimator(x)

        assert x.shape == (2, estimator.size)
        assert psds.shape == (2, self.fftlength * self.sample_rate // 2 + 1)

    def test_foreground_background_mode(self):
        psd_length = 64
        estimator = PsdEstimator(
            length=self.length,
            sample_rate=self.sample_rate,
            fftlength=self.fftlength,
        )
        background = torch.randn(
            1, 2, (psd_length + self.length) * self.sample_rate
        )
        foreground = torch.ones(
            1, 2, (psd_length + self.length) * self.sample_rate
        )
        x = torch.concat([background, foreground], dim=0)
        x, psds = estimator(x)

        assert torch.allclose(x[1], foreground[..., -estimator.size :])


class TestBatchWhitener:
    """Test suite for BatchWhitener module."""

    kernel_length = 8
    sample_rate = 2048
    inference_sampling_rate = 16
    batch_size = 32
    fduration = 1
    fftlength = 2

    def test_initialization(self):
        whitener = BatchWhitener(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
        )
        assert whitener.kernel_size == self.kernel_length * self.sample_rate
        assert whitener.stride_size == int(
            self.sample_rate / self.inference_sampling_rate
        )
        assert whitener.psd_estimator is not None
        assert whitener.whitener is not None

    def test_forward_shape(self):
        whitener = BatchWhitener(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
        )

        # Create appropriate input length
        channels = 2
        total_samples = int(
            (
                (self.batch_size - 1) * whitener.stride_size
                + whitener.kernel_size
            )
            * 2
        )
        x = torch.randn(channels, total_samples)

        kernels = whitener(x)

        assert kernels.shape == (
            self.batch_size,
            channels,
            whitener.kernel_size,
        )

    def test_return_whitened_option(self):
        whitener = BatchWhitener(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            return_whitened=True,
        )

        channels = 2
        total_samples = int(
            (
                (self.batch_size - 1) * whitener.stride_size
                + whitener.kernel_size
            )
            * 2
        )
        x = torch.randn(channels, total_samples)

        result = whitener(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        kernels, whitened = result
        assert kernels.ndim == 3
        assert whitened.squeeze().ndim == 2

    def test_with_augmentor(self):
        def simple_augmentor(x):
            return x * 2

        whitener = BatchWhitener(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
        )

        whitener_aug = BatchWhitener(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            augmentor=simple_augmentor,
        )

        channels = 2
        total_samples = int(
            (
                (self.batch_size - 1) * whitener.stride_size
                + whitener.kernel_size
            )
            * 2
        )
        x = torch.randn(channels, total_samples)

        kernels = whitener(x)
        kernels_aug = whitener_aug(x)
        assert torch.allclose(kernels_aug, kernels * 2)


class TestMultiModalPreprocessor:
    """Test suite for MultiModalPreprocessor module."""

    kernel_length = 8
    sample_rate = 2048
    inference_sampling_rate = 16
    batch_size = 32
    fduration = 1
    fftlength = 2

    def test_initialization(self):
        preprocessor = MultiModalPreprocessor(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
        )
        assert (
            preprocessor.kernel_size == self.kernel_length * self.sample_rate
        )
        assert preprocessor.stride_size == int(
            self.sample_rate / self.inference_sampling_rate
        )
        assert preprocessor.freq_mask is not None
        assert preprocessor.psd_estimator is not None
        assert preprocessor.whitener is not None

    def test_frequency_mask_creation(self):
        preprocessor = MultiModalPreprocessor(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            highpass=32,
            lowpass=256,
        )

        assert preprocessor.freq_mask.dtype == torch.bool
        assert sum(preprocessor.freq_mask) > 0

    def test_forward_shape(self):
        preprocessor = MultiModalPreprocessor(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
        )

        channels = 2
        total_samples = int(
            (
                (self.batch_size - 1) * preprocessor.stride_size
                + preprocessor.kernel_size
            )
            * 2
        )
        x = torch.randn(channels, total_samples)
        x_time, x_freq = preprocessor(x)

        num_freqs = preprocessor.freq_mask.sum()

        # Check time domain output
        assert x_time.shape == (
            self.batch_size,
            channels,
            preprocessor.kernel_size,
        )
        assert x_freq.shape == (self.batch_size, 3 * channels, num_freqs)


class TestTimeSpectrogramPreprocessor:
    """Test suite for TimeSpectrogramPreprocessor module."""

    kernel_length = 20
    sample_rate = 2048
    inference_sampling_rate = 16
    batch_size = 32
    fduration = 2
    fftlength = 2
    schedule = [[0, 16, 512], [16, 20, 2048]]  # spectrogram and timeseries
    spectrogram_shape = (64, 128)
    q = 45.6

    def test_initialization(self):
        preprocessor = TimeSpectrogramPreprocessor(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            schedule=self.schedule,
            split=True,
            q=self.q,
            spectrogram_shape=self.spectrogram_shape,
        )

        assert preprocessor.kernel_size == int(
            self.kernel_length * self.sample_rate
        )
        assert preprocessor.stride_size == int(
            self.sample_rate / self.inference_sampling_rate
        )
        assert preprocessor.psd_estimator is not None
        assert preprocessor.whitener is not None
        assert preprocessor.decimator is not None
        assert preprocessor.qtransform is not None

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            TimeSpectrogramPreprocessor(
                kernel_length=self.kernel_length,
                sample_rate=self.sample_rate,
                inference_sampling_rate=self.inference_sampling_rate,
                batch_size=self.batch_size,
                fduration=self.fduration,
                fftlength=self.fftlength,
                schedule=[[0, 16, 512]],  # only one view
                split=True,
                q=self.q,
                spectrogram_shape=self.spectrogram_shape,
            )

    def test_forward_shape(self):
        preprocessor = TimeSpectrogramPreprocessor(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            schedule=self.schedule,
            split=True,
            q=self.q,
            spectrogram_shape=self.spectrogram_shape,
        )

        channels = 2

        total_samples = int(
            (
                (self.batch_size - 1) * preprocessor.stride_size
                + preprocessor.kernel_size
            )
            * 2
        )

        x = torch.randn(channels, total_samples)

        X, X_spec = preprocessor(x)

        # expected timeseries length from schedule[1]
        ts_duration = self.schedule[1][1] - self.schedule[1][0]
        ts_rate = self.schedule[1][2]
        expected_ts_length = ts_duration * ts_rate

        # timeseries output
        assert X.shape == (
            self.batch_size,
            channels,
            expected_ts_length,
        )

        # spectrogram output
        assert X_spec.shape == (
            self.batch_size,
            channels,
            self.spectrogram_shape[0],
            self.spectrogram_shape[1],
        )

    def test_schedule_duration_matches_kernel_length(self):
        preprocessor = TimeSpectrogramPreprocessor(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            schedule=self.schedule,
            split=True,
            q=self.q,
            spectrogram_shape=self.spectrogram_shape,
        )

        # compute total scheduled duration
        schedule_duration = (
            (preprocessor.schedule[:, 1] - preprocessor.schedule[:, 0])
            .sum()
            .item()
        )

        assert schedule_duration == self.kernel_length
