import torch
from architectures.supervised import SupervisedArchitecture

from train.model.classification import AframeClassification
from train.metrics import TimeSlideAUROC

Tensor = torch.Tensor


class SupervisedAframe(AframeClassification):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X):
        return self.model(X)

    def train_step(self, batch: tuple[Tensor, Tensor, dict]) -> Tensor:
        X, y, params = batch
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

    def train_step(self, batch: tuple[Tensor, Tensor, dict]) -> Tensor:
        (X, X_fft), y, params = batch
        y_hat = self(X, X_fft)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def validation_step(self, batch, _) -> None:
        shift, (X_bg, X_bg_fft), (X_inj, X_inj_fft), params = batch

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
        self, batch: tuple[tuple[Tensor, Tensor], Tensor, dict]
    ) -> Tensor | dict[str, Tensor]:
        (X, X_spec), y, params = batch
        y_hat_X, y_hat_X_spec = self(X, X_spec)
        loss_X = torch.nn.functional.binary_cross_entropy_with_logits(
            y_hat_X, y
        )
        loss_X_spec = torch.nn.functional.binary_cross_entropy_with_logits(
            y_hat_X_spec, y
        )
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
        shift, (X_bg, X_bg_spec), (X_fg, X_fg_spec), params = batch

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
    # S4D state-space kernel parameters: trained with a small learning rate
    # and no weight decay. These names match the parameters registered by
    # ml4gw's S4DKernel; edit this tuple to change which params receive the
    # special learning rate.
    SSM_PARAM_NAMES = ("log_dt", "log_A_real", "A_imag")

    def __init__(
        self,
        arch: SupervisedArchitecture,
        *args,
        ssm_lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)
        self.save_hyperparameters("ssm_lr")

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning-rate scheduler.

        Parameters whose names appear in SSM_PARAM_NAMES are placed in their
        own optimizer group with learning rate ssm_lr and zero weight decay.
        All other parameters use learning_rate (scaled by the distributed
        world size) and weight_decay. A cosine-annealing schedule decays both
        groups from their base learning rates over the course of training.
        """
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")

        ssm_params, other_params = [], []
        for name, p in self.model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in self.SSM_PARAM_NAMES:
                ssm_params.append(p)
            else:
                other_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": other_params,
                    "lr": lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": ssm_params,
                    "lr": self.hparams.ssm_lr,
                    "weight_decay": 0.0,
                },
            ]
        )

        # Decay each group from its own base lr, preserving the lr ratio.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
