import io
import os
import shutil
from typing import Optional

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

    def get_wandb_logger(self, trainer) -> Optional[WandbLogger]:
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
        if trainer.global_rank != 0:
            return
        torch.cuda.empty_cache()
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )
        print("Checkpoiint loaded")
        device = pl_module.device
        [X], waveforms = next(iter(trainer.train_dataloader))
        print("Batch loaded")
        X = X.to(device)
        X_low, X_high, X_fft, y = trainer.datamodule.augment(X, waveforms)
        print("Augmented")
                
        trace = torch.jit.trace(module.model.to("cpu"), X_low.to("cpu"), X_high.to("cpu"), X_fft.to("cpu"))
        print("traced model")
        save_path = "/home/stevenjames.henderson/aframe/runs/multimodalv2/training/model.pt"

        #save_dir = trainer.logger.save_dir
        save_dir = os.path.dirname(save_path)
        print(f"saving to {save_dir}")
        if save_dir.startswith("s3://"):
            s3 = s3fs.S3FileSystem()
            with s3.open(f"{save_dir}/model.pt", "wb") as f:
                torch.jit.save(trace, f)

            s3.copy(self.best_model_path, f"{save_dir}/best.ckpt")
        else:
            with open(os.path.join(save_dir, "model.pt"), "wb") as f:
                torch.jit.save(trace, f)
            shutil.copy(
                self.best_model_path, os.path.join(save_dir, "best.ckpt")
            )
        print("export complete")


class SaveAugmentedBatch(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # find device module is on
            device = pl_module.device
            save_dir = trainer.logger.save_dir

            # build training batch by hand
            [X], waveforms = next(iter(trainer.train_dataloader))
            waveforms = trainer.datamodule.slice_waveforms(waveforms)
            X = X.to(device)

            X_low, X_high, X_fft, y = trainer.datamodule.augment(X, waveforms)

            # build val batch by hand
            [background, _, _], [signals] = next(
                iter(trainer.datamodule.val_dataloader())
            )
            background = background.to(device)
            signals = signals.to(device)
            X_bg, X_inj, psds = trainer.datamodule.build_val_batches(
                background, signals
            )

            if save_dir.startswith("s3://"):
                s3 = s3fs.S3FileSystem()
                with s3.open(f"{save_dir}/batch.h5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            h5file["X_low"] = X_low.cpu().numpy()
                            h5file["X_high"] = X_high.cpu().numpy()
                            h5file["X_fft"] = X_fft.cpu().numpy()
                            h5file["y"] = y.cpu().numpy()
                        s3_file.write(f.getvalue())

                with s3.open(f"{save_dir}/val_batch.h5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            h5file["X_bg"] = X_bg.cpu().numpy()
                            h5file["X_inj"] = X_inj.cpu().numpy()
                            h5file["psds"] = psds.cpu().numpy()
                        s3_file.write(f.getvalue())
            else:
                with h5py.File(os.path.join(save_dir, "batch.h5"), "w") as f:
                    f["X_low"] = X_low.cpu().numpy()
                    f["X_high"] = X_high.cpu().numpy()
                    f["X_fft"] = X_fft.cpu().numpy()
                    f["y"] = y.cpu().numpy()

                with h5py.File(
                    os.path.join(save_dir, "val_batch.h5"), "w"
                ) as f:
                    f["X_bg"] = X_bg.cpu().numpy()
                    f["X_inj"] = X_inj.cpu().numpy()
                    f["psds"] = psds.cpu().numpy()

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

class ExportOnResume(Callback):
    def on_fit_end(self, trainer, pl_module):
        # Only run once per cluster rank
        if not trainer.global_rank == 0:
            return

        ckpt_path = trainer.ckpt_path if hasattr(trainer, "ckpt_path") else None
        if ckpt_path:
            print(" Resuming from checkpoint— exporting TorchScript now.")
            ts = pl_module.to_torchscript()
            # save_path = os.path.join(trainer.log_dir, "model.pt")
            save_path = "/home/stevenjames.henderson/aframe/runs/multimodalv2/training/model.pt"
            torch.jit.save(ts, save_path)
            print(f" TorchScript model saved to {save_path}")
