import torch

from ml4gw.transforms.qtransform import SingleQTransform
from train.data.base import Tensor
from train.data.supervised.supervised import SupervisedAframeDataset


class FrequencyDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self, q: float, spectrogram_shape: list[int, int], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.q = q
        self.spectrogram_shape = spectrogram_shape

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.qtransform = SingleQTransform(
            duration=self.hparams.kernel_length,
            sample_rate=self.sample_rate,
            q=self.q,
            spectrogram_shape=self.spectrogram_shape,
        ).to(self.device)

    def augment(self, X):
        X, y = super().augment(X)
        X = self.qtransform(X)
        return X, y

    def build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor]:
        X_bg, X_fg = super().build_val_batches(background, signals)
        X_bg = self.qtransform(X_bg)
        X_fg = torch.stack([self.qtransform(X) for X in X_fg])
        return X_bg, X_fg
