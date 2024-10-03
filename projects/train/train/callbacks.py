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
from ray.train.lightning import RayTrainReportCallback

BOTO_RETRY_EXCEPTIONS = (ClientError, ConnectTimeoutError)


class WandbSaveConfig(pl.cli.SaveConfigCallback):
    """
    Override of `lightning.pytorch.cli.SaveConfigCallback` for use with WandB
    to ensure all the hyperparameters are logged to the WandB dashboard.
    """

    def save_config(self, trainer, _, stage):
        if stage == "fit":
            if isinstance(trainer.logger, WandbLogger):
                # pop off unecessary trainer args
                config = self.config.as_dict()
                config.pop("trainer")
                trainer.logger.experiment.config.update(self.config.as_dict())


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        device = pl_module.device
        [X], waveforms = next(iter(trainer.train_dataloader))
        X = X.to(device)
        X, y = trainer.datamodule.augment(X, waveforms)
        trace = torch.jit.trace(module.model.to("cpu"), X.to("cpu"))

        save_dir = trainer.logger.save_dir
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

            X, y = trainer.datamodule.augment(X, waveforms)

            # build val batch by hand
            [background, _, _], [signals] = next(
                iter(trainer.datamodule.val_dataloader())
            )
            background = background.to(device)
            signals = signals.to(device)
            X_bg, X_inj = trainer.datamodule.build_val_batches(
                background, signals
            )

            if save_dir.startswith("s3://"):
                s3 = s3fs.S3FileSystem()
                with s3.open(f"{save_dir}/batch.h5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            h5file["X"] = X.cpu().numpy()
                            h5file["y"] = y.cpu().numpy()
                        s3_file.write(f.getvalue())

                with s3.open(f"{save_dir}/val_batch.h5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            h5file["X_bg"] = X_bg.cpu().numpy()
                            h5file["X_inj"] = X_inj.cpu().numpy()
                        s3_file.write(f.getvalue())
            else:
                with h5py.File(os.path.join(save_dir, "batch.h5"), "w") as f:
                    f["X"] = X.cpu().numpy()
                    f["y"] = y.cpu().numpy()

                with h5py.File(
                    os.path.join(save_dir, "val_batch.h5"), "w"
                ) as f:
                    f["X_bg"] = X_bg.cpu().numpy()
                    f["X_inj"] = X_inj.cpu().numpy()

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


class TraceModel(RayTrainReportCallback):
    """
    Callback to trace model at the end of each Ray Tune trial epoch

    Inherit from `RayTrainReportCallback` to get access to the
    trial information that is set in its __init__
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Creates a checkpoint dir with fixed name
        tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
        os.makedirs(tmpdir, exist_ok=True)

        device = pl_module.device

        # generate sample input to infer shape,
        # making sure to augment on the device
        # where the model lives
        [X], waveforms = next(iter(trainer.train_dataloader))
        X = X.to(device)
        X, _ = trainer.datamodule.augment(X, waveforms)
        trace = torch.jit.trace(pl_module.model.to("cpu"), X.to("cpu"))

        # trace the model on cpu and then
        # move model back to original device
        pl_module.model.to(device)

        # Save trace checkpoint to local
        ckpt_path = os.path.join(tmpdir, "model.pt")
        with open(ckpt_path, "wb") as f:
            torch.jit.save(trace, f)
