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

    def forward(self, X):
        # Expect X to be a dict with keys strain, psd_low, psd_high
        return self.model(X)

    def train_step(self, batch: tuple[dict[str, Tensor], Tensor]) -> Tensor:
        X, y = batch
        y_hat = self(X)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        return self(X)

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        all_parameters = list(self.model.parameters())
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")
        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=self.hparams.weight_decay
        )

        hps = [p._optim for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s)
            for s in sorted(dict.fromkeys(frozenset(hp.items()) for hp in hps))
        ]
        for hp in hps:
            group_params = [
                p for p in all_parameters if getattr(p, "_optim", None) == hp
            ]
            optimizer.add_param_group({"params": group_params, **hp})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
