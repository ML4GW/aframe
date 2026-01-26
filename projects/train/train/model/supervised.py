import torch
from architectures.supervised import SupervisedArchitecture
from typing import Union

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
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        return self(X)


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
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

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
        metric_X:TimeSlideAUROC, 
        metric_X_spec: TimeSlideAUROC,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(arch, *args, **kwargs)

        self.metric_X = metric_X
        self.metric_X_spec = metric_X_spec

    def forward(self, X, X_spec):
        return self.model(X, X_spec)

    def score(self, X, X_spec):
        return self(X, X_spec)

    def train_step(
        self, 
        batch: tuple[tuple[Tensor, Tensor], Tensor]
    ) -> Union[Tensor, dict[str, Tensor]]:
        (X, X_spec), y = batch
        y_hat_X, y_hat_X_spec = self(X, X_spec)
        loss_X = torch.nn.functional.binary_cross_entropy_with_logits(y_hat_X, y)
        loss_X_spec = torch.nn.functional.binary_cross_entropy_with_logits(y_hat_X_spec, y)
        return {
        "loss_X": loss_X, 
        "loss_X_spec": loss_X_spec,
    }
    
    def compute_loss_fn(self, **loss):
        return 0.7 * loss["loss_X_spec"] + 0.3 * loss["loss_X"]

    def validation_step(self, batch, _) -> None:
        shift, (X_bg, X_bg_spec), (X_fg, X_fg_spec) = batch

        y_bg_X, y_bg_spec = self.score(X_bg, X_bg_spec)
        y_bg = (y_bg_X + y_bg_spec)/2

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
        y_fg = (y_fg_X + y_fg_spec) / 2

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
            "valid_auroc_avg",
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
