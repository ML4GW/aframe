from typing import Literal, Optional

from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

import torch

class ConcatResNet(SupervisedArchitecture):
    def __init__(
        self,
        in_channels: int,
        layers: List[int],
        classes: int,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[List[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
        )
        self.layers = layers
        # Initialize resnets here
        self.hlresnet = ResNet1D(
            in_channels=2,
            classes=64,
            layers=layers
        )
        self.vresnet = ResNet1D(
            classes=64,
            layers=layers,
            in_channels=1,
        )
        # Initialize linear classifier here
        self.classifier = torch.nn.Linear(128, 1)

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




















