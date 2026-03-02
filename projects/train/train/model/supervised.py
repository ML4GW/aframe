import torch
from torch.nn import functional as F
from architectures.supervised import SupervisedArchitecture

from train.model.base import AframeBase
from train.metrics import TimeSlideAUROC

Tensor = torch.Tensor


class SupervisedAframe(AframeBase):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X):
        return self.model(X)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, y = batch
        y_hat = self(X)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        return self(X)


class SupervisedAframeRegression(SupervisedAframe):
    def __init__(
        self,
        arch: SupervisedArchitecture,
        loss_weights: tuple[float, float],
        alpha: float = 20,
        sigma: float = 0.01,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)
        self.loss_weights = loss_weights
        self.alpha = alpha
        self.sigma = sigma

    def weighted_mse_loss(self, heatmap_hat, heatmap):
        weights = 1 + self.alpha * heatmap
        return (weights * (heatmap_hat - heatmap) ** 2).mean()

    def generate_heatmap(
        self, mu: Tensor, sigma: float, length: int
    ) -> Tensor:
        x = torch.arange(length, device=mu.device).float()
        sigma *= length
        heatmap = torch.exp(-((x - mu.unsqueeze(-1)) ** 2) / (2 * sigma**2))
        return heatmap

    def compute_loss_fn(self, **losses):
        return sum(
            [
                losses[key] * weight
                for key, weight in zip(
                    losses.keys(), self.loss_weights, strict=True
                )
            ]
        )

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, y, mu = batch
        mask = y.bool().squeeze()

        y_hat, heatmap_hat = self(X)
        heatmap = torch.zeros_like(
            heatmap_hat, dtype=torch.float, device=X.device
        )
        mu *= heatmap_hat.shape[-1]
        heatmap[mask] = self.generate_heatmap(
            mu, sigma=self.sigma, length=heatmap_hat.shape[-1]
        )

        losses = {
            "classifier_loss": F.binary_cross_entropy_with_logits(y_hat, y),
            "weighted_mse_loss": self.weighted_mse_loss(heatmap_hat, heatmap),
        }
        return losses

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj, mu = batch
        y_bg, heatmap_bg_hat = self.score(X_bg)

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        num_views, batch, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch, *shape)
        y_fg, heatmap_fg_hat = self.score(X_inj)

        y_fg = y_fg.view(num_views, batch)
        y_fg = y_fg.mean(0)

        # include the shift associated with this data
        # in our outputs to reconstruct background
        # timeseries at aggregation time
        self.metric.update(shift, y_bg, y_fg)

        heatmap_bg = torch.zeros_like(
            heatmap_bg_hat, dtype=torch.float, device=heatmap_bg_hat.device
        )
        mu = mu.view(num_views * batch)
        heatmap_fg = self.generate_heatmap(
            mu, self.sigma, heatmap_fg_hat.shape[-1]
        )

        valid_bg_mse = self.weighted_mse_loss(heatmap_bg_hat, heatmap_bg)
        valid_fg_mse = self.weighted_mse_loss(heatmap_fg_hat, heatmap_fg)

        metric_dict = {
            "valid_auroc": self.metric,
            "valid_bg_weighted_mse": valid_bg_mse,
            "valid_fg_weighted_mse": valid_fg_mse,
        }

        # lightning will take care of updating then
        # computing the metric at the end of the
        # validation epoch
        self.log_dict(
            metric_dict,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch,
        )

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        # scale lr by number of GPUs
        # https://arxiv.org/pdf/1706.02677.pdf
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")

        backbone_params = self.model.backbone.parameters()
        heatmap_params = self.model.heatmap_head.parameters()

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr},
                {"params": heatmap_params, "lr": lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=self.hparams.pct_lr_ramp,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


class SupervisedMultiModalAframe(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X, X_fft):
        return self.model(X, X_fft)

    def score(self, X, X_fft):
        return self(X, X_fft)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        (X, X_fft), y = batch
        y_hat = self(X, X_fft)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def validation_step(self, batch, _) -> None:
        shift, (X_bg, X_bg_fft), (X_inj, X_inj_fft) = batch

        y_bg = self.score(X_bg, X_bg_fft)

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        num_views, batch, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch, *shape)
        num_views, batch, *shape = X_inj_fft.shape
        X_inj_fft = X_inj_fft.view(num_views * batch, *shape)

        y_fg = self.score(X_inj, X_inj_fft)
        y_fg = y_fg.view(num_views, batch)
        y_fg = y_fg.mean(0)

        # include the shift associated with this data
        # in our outputs to reconstruct background
        # timeseries at aggregation time
        self.metric.update(shift, y_bg, y_fg)

        # lightning will take care of updating then
        # computing the metric at the end of the
        # validation epoch
        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )


class SupervisedTimeSpectrogramAframe(SupervisedAframe):
    def __init__(
        self,
        arch: SupervisedArchitecture,
        train_X_coeff: float,
        train_X_spec_coeff: float,
        val_X_coeff: float,
        val_X_spec_coeff: float,
        metric_X: TimeSlideAUROC,
        metric_X_spec: TimeSlideAUROC,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)

        self.metric_X = metric_X
        self.metric_X_spec = metric_X_spec

        self.train_X_coeff = train_X_coeff
        self.train_X_spec_coeff = train_X_spec_coeff
        self.val_X_coeff = val_X_coeff
        self.val_X_spec_coeff = val_X_spec_coeff

    def forward(self, X, X_spec):
        return self.model(X, X_spec)

    def score(self, X, X_spec):
        return self(X, X_spec)

    def train_step(
        self, batch: tuple[tuple[Tensor, Tensor], Tensor]
    ) -> Tensor | dict[str, Tensor]:
        (X, X_spec), y = batch
        y_hat_X, y_hat_X_spec = self(X, X_spec)
        loss_X = F.binary_cross_entropy_with_logits(y_hat_X, y)
        loss_X_spec = F.binary_cross_entropy_with_logits(y_hat_X_spec, y)
        return {
            "loss_X": loss_X,
            "loss_X_spec": loss_X_spec,
        }

    def compute_loss_fn(self, **loss):
        return (
            self.train_X_coeff * loss["loss_X"]
            + self.train_X_spec_coeff * loss["loss_X_spec"]
        )

    def validation_step(self, batch, _) -> None:
        shift, (X_bg, X_bg_spec), (X_fg, X_fg_spec) = batch

        y_bg_X, y_bg_spec = self.score(X_bg, X_bg_spec)
        y_bg = (self.val_X_coeff * y_bg_X) + (
            self.val_X_spec_coeff * y_bg_spec
        )

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction

        num_views, batch, *shape = X_fg.shape
        X_fg = X_fg.view(num_views * batch, *shape)
        num_views, batch, *shape = X_fg_spec.shape
        X_fg_spec = X_fg_spec.view(num_views * batch, *shape)

        y_fg_X, y_fg_spec = self.score(X_fg, X_fg_spec)
        y_fg_X = y_fg_X.view(num_views, batch).mean(0)
        y_fg_spec = y_fg_spec.view(num_views, batch).mean(0)
        y_fg = (self.val_X_coeff * y_fg_X) + (
            self.val_X_spec_coeff * y_fg_spec
        )

        # include the shift associated with this data
        # in our outputs to reconstruct background
        # timeseries at aggregation time
        # track for timeseries and spectrogram separately
        self.metric.update(shift, y_bg, y_fg)
        self.metric_X.update(shift, y_bg_X, y_fg_X)
        self.metric_X_spec.update(shift, y_bg_spec, y_fg_spec)

        # lightning will take care of updating then
        # computing the metric at the end of the
        # validation epoch
        # tracking metric for each data type
        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "valid_auroc_X",
            self.metric_X,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "valid_auroc_X_spec",
            self.metric_X_spec,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )


class SupervisedAframeS4(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        """
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001),
        with no weight decay.

        The rest of the model can be trained with a higher learning rate
        (e.g. 0.004, 0.01) and weight decay (if desired).
        """
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")
        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=self.hparams.weight_decay
        )

        # Add parameters with special hyperparameters
        hps = [p._optim for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s)
            for s in sorted(dict.fromkeys(frozenset(hp.items()) for hp in hps))
        ]  # Unique dicts
        for hp in hps:
            params = [
                p for p in all_parameters if getattr(p, "_optim", None) == hp
            ]
            optimizer.add_param_group({"params": params, **hp})

        # Create a lr scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
