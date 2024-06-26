import io
import os
import shutil
import tempfile
import time

import h5py
import s3fs
import torch
from botocore.exceptions import ClientError
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from ray import train


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        device = pl_module.device
        batch = next(iter(trainer.train_dataloader)).to(device)
        sample_input, _ = trainer.datamodule.augment(batch[0])

        trace = torch.jit.trace(module.model.to("cpu"), sample_input.to("cpu"))

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        if save_dir.startswith("s3://"):
            s3 = s3fs.S3FileSystem()
            with s3.open(f"{save_dir}/model.pt", "wb") as f:
                torch.jit.save(trace, f)
        else:
            with open(os.path.join(save_dir, "model.pt"), "wb") as f:
                torch.jit.save(trace, f)


class SaveAugmentedBatch(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # find device module is on
            device = pl_module.device
            save_dir = trainer.logger.log_dir or trainer.logger.save_dir

            # build training batch by hand
            X = next(iter(trainer.train_dataloader))
            X = X.to(device)
            X, y = trainer.datamodule.augment(X[0])

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
            else:
                with h5py.File(os.path.join(save_dir, "batch.h5"), "w") as f:
                    f["X"] = X.cpu().numpy()
                    f["y"] = y.cpu().numpy()

                with h5py.File(
                    os.path.join(save_dir, "val_batch.h5"), "w"
                ) as f:
                    f["X_bg"] = X_bg.cpu().numpy()
                    f["X_inj"] = X_inj.cpu().numpy()


def report_with_retries(metrics, checkpoint, retries: int = 10):
    """
    Call `train.report`, which will persist checkpoints to s3,
    retrying after any possible errors
    """
    for _ in range(retries):
        try:
            train.report(metrics=metrics, checkpoint=checkpoint)
            break
        except ClientError:
            time.sleep(5)
            continue


class AframeTrainReportCallback(Callback):
    """
    Equivalent of the RayTrainReportCallback
    (https://docs.ray.io/en/latest/train/api/doc/ray.train.lightning.RayTrainReportCallback.html)
    except saves trace instead of model weights
    """

    def __init__(self) -> None:
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = os.path.join(
            tempfile.gettempdir(), self.trial_name
        )
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Creates a checkpoint dir with fixed name
        tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
        os.makedirs(tmpdir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Trace the model
        datamodule = trainer.datamodule
        device = pl_module.device
        sample = next(iter(trainer.train_dataloader))
        sample = sample.to(device)

        sample = datamodule.augment(sample[0])
        sample_input = torch.randn(1, *sample.shape[1:])
        sample_input = sample_input.to("cpu")

        # trace the model on cpu and then move model back to original device
        trace = torch.jit.trace(pl_module.model.to("cpu"), sample_input)
        pl_module.model.to(device)

        # Save trace checkpoint to local
        ckpt_path = os.path.join(tmpdir, "model.pt")
        with open(ckpt_path, "wb") as f:
            torch.jit.save(trace, f)

        # save lightning checkpoint to local
        ckpt_path = os.path.join(tmpdir, "checkpoint.ckpt")
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = train.Checkpoint.from_directory(tmpdir)
        report_with_retries(metrics, checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        torch.distributed.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)
