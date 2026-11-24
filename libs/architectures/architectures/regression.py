from typing import Literal, Optional

import torch
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

from architectures import Architecture


class RegressionArchitecture(Architecture):
    """
    Base class for regression architectures.
    Forward pass returns a tensor of shape ``(N, num_params)``.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MultiTaskArchitecture(Architecture):
    """
    Base class for multi-task architectures.
    Forward pass returns ``(logits, param_estimates)`` where
    ``logits`` has shape ``(N, 1)`` and ``param_estimates``
    has shape ``(N, num_params)``.
    """

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class RegressionTimeDomainResNet(ResNet1D, RegressionArchitecture):
    """
    ResNet1D backbone for injection parameter regression.
    Outputs a tensor of shape ``(N, num_params)``.

    Args:
        num_params:
            Number of parameters to regress. Must match the length
            of ``param_names`` in the model config.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        num_params: int,
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
            classes=num_params,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class MultiTaskTimeDomainResNet(MultiTaskArchitecture):
    """
    Shared ResNet1D backbone with separate classification and regression heads.
    Returns ``(logits, param_estimates)`` where logits has shape ``(N, 1)``
    and param_estimates has shape ``(N, num_params)``.

    Args:
        embedding_dim:
            Dimensionality of the shared backbone's output embedding,
            i.e. the input size to both heads.
        num_params:
            Number of parameters to regress. Must match the length
            of ``param_names`` in the model config.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        num_params: int,
        layers: list[int],
        embedding_dim: int = 512,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1D(
            num_ifos,
            layers=layers,
            classes=embedding_dim,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.clf_head = torch.nn.Linear(embedding_dim, 1)
        self.reg_head = torch.nn.Linear(embedding_dim, num_params)

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(X)
        return self.clf_head(h), self.reg_head(h)
