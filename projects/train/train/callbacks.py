import io
import os
import shutil
import tempfile

import h5py
import s3fs
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from ray import train


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        datamodule = trainer.datamodule
        kernel_size = int(
            datamodule.hparams.kernel_length * datamodule.sample_rate
        )
        sample_input = torch.randn(1, datamodule.num_ifos, kernel_size)
        model = module.model.to("cpu")
        trace = torch.jit.trace(model, sample_input)

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
            device = pl_module.device
            X = next(iter(trainer.train_dataloader))
            X = X.to(device)
            X, y = trainer.datamodule.augment(X[0])
            save_dir = trainer.logger.log_dir or trainer.logger.save_dir
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
        kernel_size = int(
            datamodule.hparams.kernel_length * datamodule.sample_rate
        )
        model = pl_module.model.to("cpu")
        sample_input = torch.randn(1, datamodule.num_ifos, kernel_size)
        trace = torch.jit.trace(model, sample_input)

        # Save trace checkpoint to local
        ckpt_path = os.path.join(tmpdir, "model.pt")
        with open(ckpt_path, "wb") as f:
            torch.jit.save(trace, f)

        # save lightning checkpoint to local
        # Save checkpoint to local
        ckpt_path = os.path.join(tmpdir, "checkpoint.ckpt")
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = train.Checkpoint.from_directory(tmpdir)
        train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        torch.distributed.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)
