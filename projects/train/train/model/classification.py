from architectures import Architecture

from train.metrics import TimeSlideAUROC
from train.model.base import AframeBase
import torch

Tensor = torch.Tensor


class AframeClassification(AframeBase):
    """
    Extends AframeBase with a TimeSlideAUROC validation step.

    All detection-oriented models (supervised, autoencoder, multi-task)
    should inherit from this class.

    Args:
        arch: Architecture to train on.
        metric: TimeSlideAUROC instance used to evaluate detection performance.
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)
        self.metric = metric

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj, params = batch

        y_bg = self.score(X_bg)

        num_views, batch_size, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch_size, *shape)
        y_fg = self.score(X_inj)
        y_fg = y_fg.view(num_views, batch_size).mean(0)

        self.metric.update(shift, y_bg, y_fg)

        metric_name = self.metric.__class__.__name__
        self.log(
            f"validation/{metric_name}",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
