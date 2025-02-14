import torch
from ml4gw.spectral import truncate_inverse_power_spectrum
from ml4gw.transforms.qtransform import SingleQTransform

from train.data.base import Tensor
from train.data.supervised.supervised import SupervisedAframeDataset


class SpectrogramDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self, q: float, spectrogram_shape: list[int, int], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.q = q
        self.spectrogram_shape = spectrogram_shape

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.qtransform = SingleQTransform(
            duration=self.hparams.kernel_length,
            sample_rate=self.sample_rate,
            q=self.q,
            spectrogram_shape=self.spectrogram_shape,
        )

    def augment(self, X):
        X, y, psds = super().augment(X)
        X = self.whitener(X, psds)
        X = self.qtransform(X)
        return X, y

    def build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor]:
        X_bg, X_inj, psds = super().build_val_batches(background, signals)

        # whiten and q transform background
        X_bg = self.whitener(X_bg, psds)
        X_bg = self.qtransform(X_bg)

        # whiten and q transform each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(self.qtransform(inj))

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg


class FrequencyDomainSupervisedAframeDataset(SupervisedAframeDataset):
    # TODO: make configurable via CLI
    @property
    def taper_window(self):
        """
        Window used to taper time domain data before FFT
        """
        return torch.hann_window(self.window_size)

    @property
    def window_length(self):
        return self.hparams.kernel_length + self.hparams.fduration

    @property
    def window_size(self):
        return int(self.window_length * self.sample_rate)

    @property
    def window_scale(self):
        """
        Scale factor to ensure whitened data
        is unit variance mean zero after
        applying the taper window
        """
        df = 1 / self.window_length
        window_scale = 1 / torch.sqrt(
            torch.sum(self.taper_window**2) / len(self.taper_window)
        )
        window_scale *= torch.sqrt(torch.tensor(4 * df))
        return window_scale

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        # build tapering window and transfer to device
        window = self.taper_window * self.window_scale
        self.window = window[None][None].to(self.device)

    def whiten(self, X, psd):
        psd = truncate_inverse_power_spectrum(
            psd,
            self.hparams.fduration,
            self.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        X = X - X.mean(-1, keepdim=True)
        X = X * self.window
        freqs = torch.fft.rfftfreq(
            X.shape[-1], 1 / self.sample_rate, device=self.device
        )
        mask = freqs > self.hparams.highpass
        mask *= freqs < self.hparams.lowpass
        X = torch.fft.rfft(X, dim=-1) / self.sample_rate
        X /= torch.sqrt(psd)
        X = X[..., mask]
        return X

    def build_val_batches(self, *args, **kwargs):
        X_bg, X_inj, psds = super().build_val_batches(*args, **kwargs)

        # fft whiten and bandpass in frequency domain
        X_bg = self.whiten(X_bg, psds)
        X_inj = self.whiten(X_inj, psds)

        X_bg = torch.cat([X_bg.real, X_bg.imag], dim=-2)
        X_inj = torch.cat([X_inj.real, X_inj.imag], dim=-2)

        return X_bg, X_inj

    def augment(self, X):
        X, y, psds = super().augment(X)

        # fft whiten and bandpass in frequency domain
        X = self.whiten(X, psds)

        # split into real and imaginary parts
        X = torch.cat([X.real, X.imag], dim=1)
        return X, y
