from typing import Literal, Optional

import torch
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

class MultimodalMultiband(SupervisedArchitecture):
    def __init__(
        self,
        classes: list[int],
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[list[int]],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.num_bands = len(classes)
        resnets = [ResNet1D(num_ifos,
                            layers=layers[band],
                            classes=classes[band],
                            kernel_size=kernel_size,
                            zero_init_residual=zero_init_residual,
                            groups=groups,
                            width_per_group=width_per_group,
                            stride_type=stride_type,
                            norm_layer=norm_layer,
                           ) for band in range(self.num_bands-1)]
        resnets.append(ResNet1D(num_ifos * 3,
                                layers=layers[-1],
                                classes=classes[-1],
                                kernel_size=kernel_size,
                                zero_init_residual=zero_init_residual,
                                groups=groups,
                                width_per_group=width_per_group,
                                stride_type=stride_type,
                                norm_layer=norm_layer,
                               ))
        self.resnets = torch.nn.ModuleList(resnets)
        self.fc = torch.nn.Linear(sum(classes), 1)
    
    def forward(self, X):
        X = torch.cat(tuple(self.resnets[band](X[band]) for band in range(self.num_bands)), 1).flatten(1)
        return self.fc(X)
