from unittest.mock import MagicMock

import torch
from architectures.supervised import SupervisedDecimatedSVDNet
from train.callbacks import SVDUnfreezeCallback


def _make_model(freeze=True):
    """Create a small SVD network for testing."""
    return SupervisedDecimatedSVDNet(
        num_ifos=2,
        num_branches=2,
        n_svd=16,
        branch_hidden_dims="shallow",
        branch_embed_dim=8,
        freeze_svd=freeze,
    )


def _make_trainer_and_module(model, epoch):
    """Create mock trainer and pl_module with a real optimizer."""
    # Only optimize non-frozen params (mimics real training)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.optimizers = [optimizer]

    pl_module = MagicMock()
    pl_module.model = model

    return trainer, pl_module


class TestSVDUnfreezeCallback:
    def test_before_unfreeze_epoch(self):
        """SVD params stay frozen before unfreeze_epoch."""
        model = _make_model(freeze=True)
        callback = SVDUnfreezeCallback(unfreeze_epoch=10)
        trainer, pl_module = _make_trainer_and_module(model, epoch=5)

        callback.on_train_epoch_start(trainer, pl_module)

        for svd_layer in model.svd_layers:
            for p in svd_layer.parameters():
                assert not p.requires_grad

    def test_at_unfreeze_epoch(self):
        """At unfreeze_epoch, SVD params are unfrozen and added."""
        model = _make_model(freeze=True)
        callback = SVDUnfreezeCallback(unfreeze_epoch=10, svd_lr_factor=0.01)
        trainer, pl_module = _make_trainer_and_module(model, epoch=10)

        num_groups_before = len(trainer.optimizers[0].param_groups)
        callback.on_train_epoch_start(trainer, pl_module)

        # SVD params should now require grad
        for svd_layer in model.svd_layers:
            for p in svd_layer.parameters():
                assert p.requires_grad

        # New param group should have been added
        opt = trainer.optimizers[0]
        assert len(opt.param_groups) == num_groups_before + 1

        # Check LR of new group
        svd_group = opt.param_groups[-1]
        base_lr = opt.param_groups[0]["lr"]
        assert abs(svd_group["lr"] - base_lr * 0.01) < 1e-10

    def test_idempotent(self):
        """Calling again after unfreeze has no effect."""
        model = _make_model(freeze=True)
        callback = SVDUnfreezeCallback(unfreeze_epoch=10)
        trainer, pl_module = _make_trainer_and_module(model, epoch=10)

        callback.on_train_epoch_start(trainer, pl_module)
        num_groups_after_first = len(trainer.optimizers[0].param_groups)

        # Call again at epoch 11
        trainer.current_epoch = 11
        callback.on_train_epoch_start(trainer, pl_module)
        assert len(trainer.optimizers[0].param_groups) == num_groups_after_first

    def test_model_without_set_svd_frozen(self, capsys):
        """Model without set_svd_frozen prints warning, no crash."""
        model = MagicMock(spec=[])  # No attributes at all
        callback = SVDUnfreezeCallback(unfreeze_epoch=0)

        trainer = MagicMock()
        trainer.current_epoch = 0

        pl_module = MagicMock()
        pl_module.model = model

        callback.on_train_epoch_start(trainer, pl_module)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
