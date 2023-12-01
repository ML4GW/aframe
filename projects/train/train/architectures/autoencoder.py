from collections.abc import Callable, Sequence
from typing import Optional

import torch
from torchtyping import TensorType

from ml4gw.nn.autoencoder import ConvolutionalAutoencoder, SkipConnection
from train.architectures import Architecture

# need this for type checking
batch = channels = time = None
Tensor = TensorType["batch", "channels", "time"]
Module = Callable[[...], torch.nn.Module]


class AutoencoderArchitecture(Architecture):
    """
    Dummy class for registering available architectures
    for supervised learning problems. Supervised architectures
    are expected to return a single, real-valued logit
    corresponding to a detection statistic.
    """

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError


class ConvolutionalAutoencoder(
    ConvolutionalAutoencoder, AutoencoderArchitecture
):
    """
    Light wrapper around ml4gw convolutional autoencoder
    that ensures that
    1) `groups == num_ifos` so that information about each
        IFO doesn't leak to any of the others
    2) `decode_channels == None` so that it defaults to
        `num_ifos`
    """

    def __init__(
        self,
        num_ifos: int,
        encode_channels: Sequence[int],
        kernel_size: int,
        stride: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
        norm: Module = torch.nn.BatchNorm1d,
        skip_connection: Optional[SkipConnection] = None,
    ) -> None:
        super().__init__(
            in_channels=num_ifos,
            encode_channels=encode_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=num_ifos,
            activation=activation,
            output_activation=output_activation,
            norm=norm,
            decode_channels=None,
            skip_connection=skip_connection,
        )
