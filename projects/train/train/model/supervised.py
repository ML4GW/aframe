import torch

from train.architectures.supervised import SupervisedArchitecture
from train.model.base import AframeBase

Tensor = torch.Tensor


class SupervisedAframe(AframeBase):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X):
        if self.augmentor is not None:
            X = self.augmentor(X)
        return self.model(X)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, y = batch
        y_hat = self(X)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        return self(X)
