from typing import Literal, Optional

from architectures import Architecture
from architectures.networks import S4Model, WaveNet, Xylophone
from jaxtyping import Float
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D
from torch import Tensor
import torch
from torch import nn


class SupervisedArchitecture(Architecture):
    """
    Dummy class for registering available architectures
    for supervised learning problems. Supervised architectures
    are expected to return a single, real-valued logit
    corresponding to a detection statistic.
    """

    def forward(
        self, X: Float[Tensor, "batch channels ..."]
    ) -> Float[Tensor, " batch"]:
        raise NotImplementedError


class SupervisedTimeDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedFrequencyDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos * 2,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedTimeDomainXylophone(Xylophone, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            classes=1,
            norm_layer=norm_layer,
        )


class SupervisedTimeDomainWaveNet(WaveNet, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        res_channels: int,
        layers_per_block: int,
        num_blocks: int,
        kernel_size: int = 2,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            res_channels,
            layers_per_block,
            num_blocks,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
        )


class SupervisedSpectrogramDomainResNet(ResNet2D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedS4Model(S4Model, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        d_output: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        prenorm: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
    ) -> None:
        length = int(kernel_length * sample_rate)
        super().__init__(
            length=length,
            d_input=num_ifos,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            dt_min=dt_min,
            dt_max=dt_max,
            lr=lr,
        )


class SupervisedMultiModalResNet(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.
    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """

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
        self.time_domain_resnet = ResNet1D(
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

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.classifier = torch.nn.Linear(time_classes + freq_classes, 1)

    def forward(self, X, X_fft):
        time_domain_output = self.time_domain_resnet(X)
        freq_domain_output = self.freq_psd_resnet(X_fft)
        concat = torch.cat([time_domain_output, freq_domain_output], dim=-1)
        return self.classifier(concat)


class SupervisedTimeSpectrogramResNet(SupervisedArchitecture):
    """
    Spectrogram and Time Domain ResNet that processes a combination of
    timeseries and spectrogram image data.
    """

    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        spec_classes: int,
        time_layers: list[int],
        spec_layers: list[int],
        time_kernel_size: int = 3,
        spec_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        time_norm_layer: Optional[NormLayer] = None,
        spec_norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=time_norm_layer,
        )

        self.spectrogram_resnet = ResNet2D(
            in_channels=num_ifos,
            layers=spec_layers,
            classes=spec_classes,
            kernel_size=spec_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=spec_norm_layer,
        )

    def forward(self, X, X_spec):
        time_domain_output = self.time_domain_resnet(X)
        spec_domain_output = self.spectrogram_resnet(X_spec)
        return time_domain_output, spec_domain_output


class AddCoords1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, length)
        """
        batch_size, _, length = x.size()
        # Create a linear ramp from -1 to 1
        # [length] -> [1, 1, length]
        pos = torch.linspace(-1, 1, length, device=x.device, dtype=x.dtype)
        pos = pos.view(1, 1, length)
        # Expand to match batch size: [batch_size, 1, length]
        pos = pos.expand(batch_size, -1, -1)
        # Concatenate the coordinate channel to the input
        # Result shape: [batch_size, channels + 1, length]
        return torch.cat([x, pos], dim=1)


class CoordConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.add_coords = AddCoords1d()
        # Note: in_channels + 1 because of the extra coordinate channel
        self.conv = nn.Conv1d(in_channels + 1, out_channels, **kwargs)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        return x


class ClippedLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, max_val=1.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.max_val = max_val

    def forward(self, x):
        x = nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        return torch.clamp(x, max=self.max_val)


def _convert_padding_mode(model, mode="reflect"):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            module.padding_mode = mode
            module.stride = 1
    return model


class SupervisedTimeDomainRegression(SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        final_activation: str | None = "sigmoid",
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1D(
            num_ifos,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.backbone.conv1 = CoordConv1d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_channels = self.backbone.residual_layers[-1][-1].conv2.out_channels
        self.dilated_layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=8,
                dilation=8,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=32,
                dilation=32,
            ),
            nn.ReLU(),
        )

        activation = (
            nn.Sigmoid()
            if final_activation == "sigmoid"
            else ClippedLeakyReLU()
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(
                self.backbone.fc.in_features, 128, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1, bias=False),
            activation,
        )
        _convert_padding_mode(self)

    def _forward_impl(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        for layer in self.backbone.residual_layers:
            x = layer(x)

        return x

    def forward(self, x):
        x = self._forward_impl(x)
        heatmap = self.heatmap_head(self.dilated_layer(x))
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.backbone.fc(x)
        return y, heatmap.squeeze(1)
