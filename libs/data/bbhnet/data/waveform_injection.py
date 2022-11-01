import torch

from ml4gw.transforms.injection import RandomWaveformInjection


class BBHNetWaveformInjection(RandomWaveformInjection):
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return X, y

        X, idx, _ = super().forward(X)
        y[idx] = 1
        return X, y
