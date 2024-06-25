import torch

from ml4gw.transforms.qtransform import SingleQTransform
from train.data.base import Tensor
from train.data.supervised.supervised import SupervisedAframeDataset


class SpectrogramSupervisedAframeDataset(SupervisedAframeDataset):
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


class FrequencyDomainAframeDataset(SupervisedAframeDataset):
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)

        # build tapering window
        window_length = self.hparams.kernel_length + self.hparams.fduration
        window_size = int(window_length * self.sample_rate)
        window = torch.hann_window(window_size)
        df = 1 / window_length
        window_scale = torch.sum(window**2) / len(window)
        window /= torch.sqrt(window_scale)
        window *= torch.sqrt(4 * df)
        self.window = window[None][None].to(self.device)

    def build_val_batches(self, *args, **kwargs):
        X_bg, X_inj, psds = super().build_val_batches(*args, **kwargs)

        # fft and whiten in frequeny domain
        X_bg = X_bg * self.window
        X_bg = torch.fft.rfft(X_bg, dim=-1) / self.sample_rate
        X_bg /= torch.sqrt(psds)

        X_inj = X_inj * self.window
        X_inj = torch.fft.rfft(X_bg, dim=-1) / self.sample_rate
        X_inj /= torch.sqrt(psds)

        X_bg = torch.cat([X_bg.real, X_bg.imag], dim=-2)
        X_inj = torch.cat([X_inj.real, X_inj.imag], dim=-2)

        return X_bg, X_inj

    def augment(self, X):
        X, y, psds = super().augment(X)

        # fft and whiten in frequency domain
        X *= self.window
        X = torch.fft.rfft(X, dim=-1) / self.sample_rate
        X /= torch.sqrt(psds)

        # split into real and imaginary parts
        X = torch.cat([X.real, X.imag], dim=1)
        return X, y
