from ml4gw.transforms.qtransform import SingleQTransform
from projects.train.train.data.base import Tensor
from train.data.supervised import SupervisedAframeDataset


class FrequencyDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        *args,
        qtransform: SingleQTransform,
        num_f_bins: int,
        num_t_bins: int,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.qtransform = qtransform
        self.num_f_bins = num_f_bins
        self.num_t_bins = num_t_bins

    def augment(self, X):
        X, y = super().augment(X)
        X = self.qtransform(X, self.num_f_bins, self.num_t_bins)
        return X, y

    def build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor]:
        X_bg, X_fg = super().build_val_batches(background, signals)
        X_bg = self.qtransform(X_bg, self.num_f_bins, self.num_t_bins)
        X_fg = self.qtransform(X_fg, self.num_f_bins, self.num_t_bins)
        return X_bg, X_fg
