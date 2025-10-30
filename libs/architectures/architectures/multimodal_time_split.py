import torch
import torch.nn as nn
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import ResNet1D, NormLayer
from typing import Optional, Literal


class MultimodalTimeSplitSupervisedArchitecture(SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        low_time_classes: int,
        high_time_classes: int,
        freq_classes: int,
        low_time_layers: list[int],
        high_time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()

        # Time-domain ResNets
        self.strain_low_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=low_time_layers,
            classes=low_time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.strain_high_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=high_time_layers,
            classes=high_time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        # Frequency-domain ResNet
        freq_input_channels = int(num_ifos * 3)
        self.fft_resnet = ResNet1D(
            in_channels=freq_input_channels,
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        embed_dim = high_time_classes + low_time_classes + freq_classes
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(
        self, x_low: torch.Tensor, x_high: torch.Tensor, x_fft: torch.Tensor
    ):
        low_out = self.strain_low_resnet(x_low)
        high_out = self.strain_high_resnet(x_high)
        fft_out = self.fft_resnet(x_fft)

        features = torch.cat([low_out, high_out, fft_out], dim=-1)
        return self.classifier(features)