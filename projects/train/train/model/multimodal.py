import torch
from architectures.supervised import SupervisedArchitecture
from train.model.base import AframeBase
import torch.nn.functional as F

Tensor = torch.Tensor

class MultimodalAframe(AframeBase):
    def __init__(
        self,
        arch: SupervisedArchitecture,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, x_low: Tensor, x_high: Tensor, x_fft: Tensor) -> Tensor:
        return self.model(x_low, x_high, x_fft)

    def train_step(self, batch: tuple) -> Tensor:
        print(" training Step ~")
        # Unpack depending on number of elements
        if len(batch) == 4:
            X_low, X_high, X_fft, y = batch
        else:
            raise ValueError(f"Unexpected batch format in train_step: {len(batch)} elements")

        y_hat = self((X_low, X_high, X_fft))
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        X_low, X_high, X_fft = X
        return self.model(X_low, X_high, X_fft)

    def validation_step(self, batch, batch_idx):
        try:
            shift, X_bg, X_inj = batch
        except ValueError:
            shift, X_bg_low, X_bg_high, X_bg_fft, X_fg_low, X_fg_high, X_fg_fft, *_ = batch
            X_bg = (X_bg_low, X_bg_high, X_bg_fft)
            X_inj = (X_fg_low, X_fg_high, X_fg_fft)

        # Score background
        y_bg = self.score(X_bg)

        # Score injected signals (num_views, batch, ...)
        num_views, batch, *_ = X_inj[0].shape  # assume all modalities same shape
        X_inj = tuple(x.view(num_views * batch, *x.shape[2:]) for x in X_inj)
        y_fg = self.score(X_inj)
        y_fg = y_fg.view(num_views, batch).mean(0)

        self.metric.update(shift, y_bg, y_fg)

        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

