from typing import List
import warnings
import torch
import torch.nn.functional as F
from architectures import Architecture


from train.model.base import AframeBase

Tensor = torch.Tensor


class SupervisedRegressionAframe(AframeBase):
    """
    Supervised model that predicts injection parameters via regression only,
    with no classification head or detection objective.

    Designed to be used with ``waveform_prob=1.0`` so that every training
    sample contains an injection. Non-injected samples (NaN params) are
    masked out of the loss gracefully, but they are wasteful.

    For validation set ``num_valid_views=1`` in the data config — multiple
    views are only meaningful for averaging detection scores, not for
    parameter recovery.

    The architecture's forward pass must return ``param_estimates`` of
    shape ``(N, len(param_names))``.

    Args:
        arch:
            Architecture whose forward pass returns param_estimates.
        param_names:
            Ordered list of parameter names to regress on. Must match
            the output ordering of the architecture's regression head.
    """

    def __init__(
        self,
        arch: Architecture,
        param_names: List[str],
        **kwargs,
    ):
        super().__init__(arch=arch, **kwargs)
        self.param_names = param_names

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)

    def score(self, X: Tensor) -> Tensor:
        return self(X)

    def train_step(self, batch):
        X, y, params = batch
        mask = ~torch.isnan(next(iter(params.values())))

        if not mask.any():
            warnings.warn(
                "All samples in batch have NaN parameters;"
                "skipping regression step.",
                stacklevel=2,
            )
            return torch.zeros(1, device=X.device, requires_grad=True)

        targets = torch.stack(
            [params[k][mask] for k in self.param_names], dim=1
        )
        return F.mse_loss(self(X[mask]), targets)

    def validation_step(self, batch, _):
        _, _, X_inj, params = batch
        num_views, N, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * N, *shape)

        param_estimates = self(X_inj)

        for i, name in enumerate(self.param_names):
            targets = params[name].repeat(num_views)
            mae = F.l1_loss(param_estimates[:, i], targets)
            self.log(
                f"validation/mae_{name}",
                mae,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
