from projects.train.train.data.base import Tensor
from train.data.supervised.supervised import SupervisedAframeDataset


class FrequencyDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def augment(self, X):
        X, y = super().augment(X)
        return X, y

    def build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor]:
        X_bg, X_fg = super().build_val_batches(background, signals)
        return X_bg, X_fg
