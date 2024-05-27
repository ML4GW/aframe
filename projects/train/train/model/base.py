import logging
from typing import Optional, Sequence, Union

import lightning.pytorch as pl
import ray
import torch
from architectures import Architecture

from train.callbacks import ModelCheckpoint, SaveAugmentedBatch
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
        weight_decay: float = 0.0,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir;
        self.model = arch
        self.metric = metric
        self.save_hyperparameters(ignore=["arch", "metric"])
        self.verbose = verbose
        self._logger = self.get_logger()

    def get_logger(self):
        logger_name = "AframeModel"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        return logger

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

    # define some hacky callbacks that can be
    # used during the `self.score` method of
    # downstream child classes to record more
    # detailed plots during the course of training
    def on_validation_epoch_start(self):
        self.validating = True

    def on_validation_epoch_end(self):
        self.validating = False

    def on_validation_score(self, *tensors):
        for cb in self.trainer.callbacks:
            if hasattr(cb, "on_validation_score"):
                cb.on_validation_score(self.trainer, self, *tensors)

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

        # TODO: maybe check if our model has a .loss
        # attribute and if so include it as a loss
        # term, this way models can do arbitrary things
        # to penalize themselves if desired. Should probably
        # just be added to whatever pops out of
        # self.compute_loss_fn, under the assumption that the
        # model has applied the appropriate scale to this
        # value. More complicated functionality can be
        # achieved by subclasses in self.train_step.

        # if our train step returned a dictionary of losses,
        # log them all separately then combine them into a
        # single loss via `compute_loss_fn`
        if isinstance(loss, dict):
            for name, value in loss.items():
                self.log(
                    name,
                    value.mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            loss = self.compute_loss_fn(**loss)

        loss = loss.mean()
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
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []
        callbacks.append(SaveAugmentedBatch())

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = ModelCheckpoint(
            monitor="valid_auroc",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
        )

        if not ray.is_initialized():
            callbacks.append(checkpoint)

        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStopping(
                monitor="valid_auroc",
                patience=self.hparams.patience,
                mode="max",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        # scale lr by number of GPUs
        # https://arxiv.org/pdf/1706.02677.pdf
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=self.hparams.pct_lr_ramp,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = dict(scheduler=scheduler, interval="step")
        return dict(optimizer=optimizer, lr_scheduler=scheduler_config)
