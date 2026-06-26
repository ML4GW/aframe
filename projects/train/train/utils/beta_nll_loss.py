from torch import nn
import torch


class BetaNLLLoss(nn.Module):
    """β-NLL loss (Seitzer et al. 2022, https://arxiv.org/abs/2203.09168).

    Standard GaussianNLL has a degenerate minimum where inflating var drives
    the mean gradient to zero.  β-weighting prevents this:

        L_β = sg(var)^β · L_NLL
        ∂L_β/∂mean = sg(var)^(β−1) · (mean − y)

    β=0 → standard NLL (degenerate); β=0.5 → recommended; β=1 → MSE gradient.
    """

    def __init__(self, beta: float = 0.5, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta
        self.reduction = reduction

    def forward(
        self, mean: torch.Tensor, target: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        nll = 0.5 * (torch.log(var) + (mean - target) ** 2 / var)
        if self.beta > 0.0:
            nll = nll * var.detach().pow(self.beta)
        return nll.mean() if self.reduction == "mean" else nll.sum()
