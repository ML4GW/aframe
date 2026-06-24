import io
import os
import shutil

import h5py
import s3fs
import torch
from botocore.exceptions import ClientError, ConnectTimeoutError
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm

BOTO_RETRY_EXCEPTIONS = (ClientError, ConnectTimeoutError)


class WandbSaveConfig(pl.cli.SaveConfigCallback):
    """
    Override of `lightning.pytorch.cli.SaveConfigCallback` for use with WandB
    to ensure all the hyperparameters are logged to the WandB dashboard.
    """

    def get_wandb_logger(self, trainer) -> WandbLogger | None:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    def save_config(self, trainer, _, stage) -> None:
        wandb_logger = self.get_wandb_logger(trainer)
        if stage == "fit" and wandb_logger is not None:
            # pop off unecessary trainer args
            config = self.config.as_dict()
            config.pop("trainer")
            wandb_logger.experiment.config.update(config)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        # Restore the best checkpoint's weights for export.
        # torch>=2.6 defaults torch.load to weights_only=True,
        # which rejects the pickled distribution objects
        # (e.g. ml4gw.distributions.Cosine) in the saved hyperparameters.
        # Set weights_only=False and load the state_dict into pl_module.
        ckpt = torch.load(
            self.best_model_path, map_location="cpu", weights_only=False
        )
        pl_module.load_state_dict(ckpt["state_dict"])

        device = pl_module.device
        # Handle the case of loading training waveforms from disk
        if trainer.datamodule.waveforms_from_disk:
            [X], waveforms = next(iter(trainer.train_dataloader))
            X = X.to(device)
            waveforms = waveforms.to(device)
            X, _ = trainer.datamodule.inject(X, waveforms)
        else:
            [X] = next(iter(trainer.train_dataloader))
            X = X.to(device)
            X, _ = trainer.datamodule.inject(X)

        example = X if isinstance(X, tuple) else (X,)
        example = tuple(x.cpu() for x in example)
        pl_module.model.eval()
        # Dim.AUTO allows the batch size to be dynamic
        dynamic_shapes = tuple({0: torch.export.Dim.AUTO} for _ in example)
        exported = torch.export.export(
            pl_module.model.to("cpu"), example, dynamic_shapes=dynamic_shapes
        )

        save_dir = trainer.logger.save_dir
        if save_dir.startswith("s3://"):
            s3 = s3fs.S3FileSystem()
            with s3.open(f"{save_dir}/model_exported.pt2", "wb") as f:
                torch.export.save(exported, f)
            s3.copy(self.best_model_path, f"{save_dir}/best.ckpt")
        else:
            with open(os.path.join(save_dir, "model_exported.pt2"), "wb") as f:
                torch.export.save(exported, f)
            shutil.copy(
                self.best_model_path, os.path.join(save_dir, "best.ckpt")
            )


class SaveAugmentedBatch(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # find device module is on
            device = pl_module.device
            save_dir = trainer.logger.save_dir

            # build training batch by hand
            # Handle the case of loading training waveforms from disk
            if trainer.datamodule.waveforms_from_disk:
                [X], waveforms = next(iter(trainer.train_dataloader))
                X = X.to(device)
                waveforms = waveforms.to(device)
                X, y = trainer.datamodule.inject(X, waveforms)
            else:
                [X] = next(iter(trainer.train_dataloader))
                X = X.to(device)
                X, y = trainer.datamodule.inject(X)
            # If X is not a tuple, make it one for consistency
            # of format for saving to file below
            if not isinstance(X, tuple):
                X = (X,)

            # build val batch by hand
            [background, _, _], [signals] = next(
                iter(trainer.datamodule.val_dataloader())
            )
            background = background.to(device)
            signals = signals.to(device)
            X_bg, X_inj = trainer.datamodule.build_val_batches(
                background, signals
            )
            # Make background and injected validation data into
            # tuples for consistency if necessary
            if not isinstance(X_bg, tuple):
                X_bg = (X_bg,)
            if not isinstance(X_inj, tuple):
                X_inj = (X_inj,)

            if save_dir.startswith("s3://"):
                s3 = s3fs.S3FileSystem()
                with s3.open(f"{save_dir}/batch.hdf5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            for i, x in enumerate(X):
                                h5file[f"input_{i}"] = x.cpu().numpy()
                            h5file["y"] = y.cpu().numpy()
                        s3_file.write(f.getvalue())

                with s3.open(f"{save_dir}/val_batch.hdf5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            for i, (bg, inj) in enumerate(
                                zip(X_bg, X_inj, strict=True)
                            ):
                                h5file[f"X_bg_{i}"] = bg.cpu().numpy()
                                h5file[f"X_inj_{i}"] = inj.cpu().numpy()
                        s3_file.write(f.getvalue())
            else:
                with h5py.File(os.path.join(save_dir, "batch.hdf5"), "w") as f:
                    for i, x in enumerate(X):
                        f[f"input_{i}"] = x.cpu().numpy()
                    f["y"] = y.cpu().numpy()

                with h5py.File(
                    os.path.join(save_dir, "val_batch.hdf5"), "w"
                ) as f:
                    for i, (bg, inj) in enumerate(
                        zip(X_bg, X_inj, strict=True)
                    ):
                        f[f"X_bg_{i}"] = bg.cpu().numpy()
                        f[f"X_inj_{i}"] = inj.cpu().numpy()

            # while we're here let's log the wandb url
            # associated with the run
            maybe_wandb_logger = trainer.loggers[-1]
            if isinstance(maybe_wandb_logger, pl.loggers.WandbLogger):
                url = maybe_wandb_logger.experiment.url
                if save_dir.startswith("s3://"):
                    with s3.open(f"{save_dir}/wandb_url.txt", "wb") as s3_file:
                        s3_file.write(url.encode())
                else:
                    with open(
                        os.path.join(save_dir, "wandb_url.txt"), "w"
                    ) as f:
                        f.write(url)


class GradientTracker(Callback):
    def __init__(self, norm_type: int = 2):
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        total_norm = norms[f"grad_{float(self.norm_type)}_norm_total"]
        self.log(f"grad_norm_{self.norm_type}", total_norm)
