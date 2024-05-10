from typing import Literal, Optional

from torch import Tensor
from torchtyping import TensorType

from aframe.architectures import Architecture
from aframe.architectures.networks import WaveNet, Xylophone
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D

# need this for type checking
batch = channels = None


class SupervisedArchitecture(Architecture):
    """
    Dummy class for registering available architectures
    for supervised learning problems. Supervised architectures
    are expected to return a single, real-valued logit
    corresponding to a detection statistic.
    """

    # TODO: torchtyping doesn't support ellipsis at the
    # end, but I want to use that here because we could
    # have either one or two dimensions at the end
    # depending on whether we're in the time or time-frequency
    # domain. Moving ml4gw to jaxtyping should be able to
    # handle this.
    def forward(
        self, X: Tensor  # Type["batch", "channels", ...]
    ) -> TensorType["batch", 1]:
        raise NotImplementedError


class SupervisedTimeDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
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


class SupervisedTimeDomainXylophone(Xylophone, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
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
        res_channels: int,
        layers_per_block: int,
        num_blocks: int,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            res_channels,
            layers_per_block,
            num_blocks,
            norm_layer=norm_layer,
        )


class SupervisedFrequencyDomainResNet(ResNet2D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
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
