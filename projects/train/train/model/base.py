from typing import Optional, Sequence, Union

import lightning.pytorch as pl
import torch

from train.architectures import Architecture
from train.metrics import TimeSlideAUROC

Tensor = torch.Tensor


class AframeBase(pl.LightningModule):
    """
    Args:
        arch: Architecture to train on
        metric: Metric used for evaluation
        learning_rate:
            Hyperparameter controlling size of gradient steps
            during training
        pct_lr_ramp:
            Fraction of number of training epochs over which
            learning rate will ramp up to its specified value
        patience:
            Number of epochs to wait for an increase in
            validation AUROC before terminating training.
            If left as `None`, will never terminate
            training early
        save_top_k_models:
            Maximum number of best-performing model checkpoints
            to keep during training
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        learning_rate: float,
        pct_lr_ramp: float,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        self.metric = metric
        self.save_hyperparameters(ignore=["arch", "metric"])

    def forward(self, X: Tensor) -> Tensor:
        """
        Override this method to dictate how the outputs
        of the neural network component of your model
        are generated. This is distinct from `self.score`
        in that the latter might include post-processing
        steps used to produce a single detection statistic
        value. E.g. for autoencoders, `score` might include
        computing some reconstruction loss.
        """
        raise NotImplementedError

    def train_step(self, batch: Tensor) -> Union[Tensor, dict[str, Tensor]]:
        """
        Override this method to dictate how your model
        produces loss(es) to be optimized during training.
        Can either return a single tensor specifying batch-level
        losses, or a dictionary mapping from names of different
        loss terms to tensors. In the latter case, each loss term
        will be logged separately and they'll be combined together
        using `compute_loss_fn`, which should be overridden in that case.
        """
        raise NotImplementedError

    def score(self, X: Tensor) -> Tensor:
        """
        Override this method to produce a detection
        statistics for a batch of inputs.
        """
        raise NotImplementedError

    def compute_loss_fn(self, **losses):
        """
        Override this method if your train step returns
        a dictionary of losses to aggregate them into
        a single loss that gets optimized.
        """
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        loss = self.train_step(batch)

        # if our train step returned a dictionary of losses,
        # log them all separately then combine them into a
        # single loss via `compute_loss_fn`
        if isinstance(loss, dict):
            for name, value in loss.items():
                self.log(
                    f"{name}_loss",
                    value.mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            loss = self.compute_loss_fn(**loss)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj = batch
        y_bg = self.score(X_bg)

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        num_views, batch, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch, *shape)
        y_fg = self.score(X_inj)
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

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=self.metric_name,
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
        )
        callbacks = [checkpoint]
        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStop(
                monitor=self.metric_name,
                patience=self.hparams.patience,
                mode="max",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=self.hparams.pct_lr_ramp,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
