from ml4gw.transforms.spectrogram import MultiResolutionSpectrogram
from projects.train.train.data.base import Tensor
from train.data.supervised import SupervisedAframeDataset


class FrequencyDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self, *args, spectrogram: MultiResolutionSpectrogram, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spectrogram = spectrogram

    def augment(self, X):
        X, y = super().augment(X)
        X = self.spectrogram(X)
        return X, y

    def build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor]:
        X_bg, X_fg = super().build_val_batches(background, signals)
        X_bg = self.spectrogram(X_bg)
        X_fg = self.spectrogram(X_fg)
        return X_bg, X_fg
