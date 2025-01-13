from typing import Literal, Optional

from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

import torch

class ConcatResNet(SupervisedArchitecture):
    def __init__(
        self,
        v_dim: int,
        v_layers: list[int],
        hl_dim: int,
        hl_layers: list[int],
        num_ifos: int,
        layers: list[int],
        sample_rate: float,
        kernel_length: float,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
        )
        # Initialize resnets here
        self.hlresnet = ResNet1D(
            in_channels=2,
            classes=hl_dim,
            layers=hl_layers,
        )
        self.vresnet = ResNet1D(
            classes=v_dim,
            layers=v_layers,
            in_channels=1,
        )
        # Initialize linear classifier here
        self.classifier = torch.nn.Linear(hl_dim+v_dim, 1)

    def forward(self, X):
        # Extract hl data and v data from X
        hl = X[:, :2, :]
        v = X[:, 2:, :]
        # Pass hl and v through resnets
        hl = self.hlresnet(hl)
        v = self.vresnet(v)
        # Concatenate hl and v outputs
        concat = torch.concat([hl, v], dim=-1)
        # Pass concatenated output through linear classifier
        outputs = self.classifier(concat)

        return outputs




















