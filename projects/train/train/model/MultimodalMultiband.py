import torch
from architectures.supervised import SupervisedArchitecture

from train.model.base import AframeBase

Tensor = torch.Tensor


class MultimodalMultibandModule(AframeBase):
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

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj = batch
        y_bg = self.score(X_bg)

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        X_flattened = tuple()
        for inj in X_inj:
            num_views, batch, *shape = inj.shape
            inj = inj.view(num_views * batch, *shape)
            X_flattened = X_flattened + (inj,)
            
        y_fg = self.score(X_flattened)
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