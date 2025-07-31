import torch
from architectures.supervised import SupervisedArchitecture
from train.model.base import AframeBase

Tensor = torch.Tensor


class MultimodalAframe(AframeBase):
    def __init__(
        self,
        arch: SupervisedArchitecture,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, x_low: Tensor, x_high: Tensor, x_fft: Tensor) -> Tensor:
        return self.model(x_low, x_high, x_fft)

    def train_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        x_low, x_high, x_fft, y = batch
        y_hat = self(x_low, x_high, x_fft)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def validation_step(self, batch, batch_idx):
        (
            shift,
            x_bg_low, x_bg_high, x_bg_fft,
            x_fg_low, x_fg_high, x_fg_fft,
            _
        ) = batch

        if isinstance(shift, torch.Tensor):
            if shift.numel() > 1:
                shift = shift[0].item()
            else:
                shift = shift.item()
        else:
            shift = float(shift)

        y_hat_bg = self.score(x_bg_low, x_bg_high, x_bg_fft)
        y_hat_fg = self.score(x_fg_low, x_fg_high, x_fg_fft)

        y_bg = torch.zeros_like(y_hat_bg)
        y_fg = torch.ones_like(y_hat_fg)

        y_hat = torch.cat([y_hat_bg, y_hat_fg], dim=0)
        y = torch.cat([y_bg, y_fg], dim=0)

        self.metric.update(shift, y_hat_bg, y_hat_fg)
        self.log("valid_auroc", self.metric.compute(), prog_bar=True, on_epoch=True)

    def score(self, x_low: Tensor, x_high: Tensor, x_fft: Tensor):
        return self(x_low, x_high, x_fft)

