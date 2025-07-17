import torch
import torch.nn as nn
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import ResNet1D, NormLayer
from typing import Optional, Literal


class MultimodalSupervisedArchitecture(SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        freq_classes: int,
        time_layers: list[int],
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

        # Time-domain ResNet
        self.strain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        # Frequency-domain ResNets
        freq_input_channels = int(num_ifos * 3)
        self.psd_low_resnet = ResNet1D(
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
        self.psd_high_resnet = ResNet1D(
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

        # Final classifier
        embed_dim = time_classes + 2 * freq_classes
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, X):
        strain = X["strain"]
        psd_low = X["psd_low"]
        psd_high = X["psd_high"]

        strain_out = self.strain_resnet(strain)
        low_out = self.psd_low_resnet(psd_low)
        high_out = self.psd_high_resnet(psd_high)

        features = torch.cat([strain_out, low_out, high_out], dim=-1)
        return self.classifier(features)
