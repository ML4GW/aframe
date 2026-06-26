from typing import List, Optional
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from architectures import Architecture


from train.model.base import AframeBase
from train.utils.beta_nll_loss import BetaNLLLoss

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


class GaussianNLLRegressionAframe(SupervisedRegressionAframe):
    """
    Regression model that predicts each parameter's value AND its uncertainty,
    trained with a (beta-weighted) Gaussian negative-log-likelihood loss.

    Where ``SupervisedRegressionAframe`` predicts a single value per parameter
    and trains with MSE, this model predicts two numbers per parameter: a mean
    and a raw variance. The raw variance is passed through ``Softplus`` to keep
    it positive. The architecture's forward must therefore return shape
    ``(N, 2 * len(param_names))`` with the means in the first half and the raw
    variances in the second half.

    The detection score is the negative mean predicted variance, so a
    confident (low-uncertainty) prediction scores high.

    Args:
        arch:
            Architecture whose forward returns ``(N, 2 * len(param_names))``.
        param_names:
            Ordered parameter names to regress on.
        beta_nll:
            beta for the beta-NLL loss (Seitzer et al. 2022). ``0`` gives plain
            Gaussian NLL, ``0.5`` is the recommended default, ``1`` recovers an
            MSE-like gradient.
        y_mean, y_std:
            Optional per-parameter normalization applied to the targets (and
            inverted on the predictions when reporting). Length
            ``len(param_names)``; default to zero-mean / unit-std (no-op).
    """

    def __init__(
        self,
        arch: Architecture,
        param_names: List[str],
        beta_nll: float = 0.5,
        y_mean: Optional[List[float]] = None,
        y_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(arch=arch, param_names=param_names, **kwargs)
        self.n_vars = len(param_names)
        self.var_activation = nn.Softplus()
        self.criterion = BetaNLLLoss(beta=beta_nll)

        _y_mean = (
            torch.tensor(y_mean, dtype=torch.float32)
            if y_mean is not None
            else torch.zeros(self.n_vars)
        )
        _y_std = (
            torch.tensor(y_std, dtype=torch.float32)
            if y_std is not None
            else torch.ones(self.n_vars)
        )
        self.register_buffer("y_mean", _y_mean)
        self.register_buffer("y_std", _y_std)

    def _split(self, outputs: Tensor) -> tuple[Tensor, Tensor]:
        """Split the network output into (mean, positive variance)."""
        mean = outputs[:, : self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars :])
        return mean, var

    def _normalize(self, y: Tensor) -> Tensor:
        return (y - self.y_mean) / self.y_std

    def score(self, X: Tensor) -> Tensor:
        # detection score: lower predicted uncertainty -> higher score
        _, var = self._split(self(X))
        return -var.mean(dim=-1)

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
        mean, var = self._split(self(X[mask]))
        return self.criterion(mean, self._normalize(targets), var)

    def validation_step(self, batch, _):
        _, _, X_inj, params = batch
        num_views, N, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * N, *shape)

        mean, var = self._split(self(X_inj))
        # convert back to physical units for reporting
        mean_phys = mean * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var) * self.y_std

        for i, name in enumerate(self.param_names):
            targets = params[name].repeat(num_views)
            self.log(
                f"validation/mae_{name}",
                F.l1_loss(mean_phys[:, i], targets),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"validation/sigma_{name}",
                sigma_phys[:, i].mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
