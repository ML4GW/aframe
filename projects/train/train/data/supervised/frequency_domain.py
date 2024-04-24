import torch

from ml4gw.transforms.qtransform import SingleQTransform
from projects.train.train.data.base import Tensor
from train.data.supervised import SupervisedAframeDataset


class FrequencyDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, qtransform: SingleQTransform, **kwargs):
        super().__init__(*args, **kwargs)
        self.qtransform = qtransform

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
