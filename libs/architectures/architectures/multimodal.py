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

        # Time-domain ResNets
        self.strain_low_resnet = ResNet1D(
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

        self.strain_high_resnet = ResNet1D(
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

        embed_dim = 2 * time_classes + freq_classes
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor, x_fft: torch.Tensor):
        if x_low.ndim == 4:
            # [num_views, batch, C, L] -> [num_views * batch, C, L]
            x_low = x_low.flatten(0, 1)
            x_high = x_high.flatten(0, 1)
            x_fft = x_fft.flatten(0, 1)

        assert x_low.shape[1] == self.strain_low_resnet.conv1.in_channels, f"x_low has wrong shape: {x_low.shape}"
        assert x_high.shape[1] == self.strain_high_resnet.conv1.in_channels, f"x_high has wrong shape: {x_high.shape}"
        assert x_fft.shape[1] == self.fft_resnet.conv1.in_channels, f"x_fft has wrong shape: {x_fft.shape}"

        low_out = self.strain_low_resnet(x_low)
        high_out = self.strain_high_resnet(x_high)
        fft_out = self.fft_resnet(x_fft)

        features = torch.cat([low_out, high_out, fft_out], dim=-1)
        return self.classifier(features)

