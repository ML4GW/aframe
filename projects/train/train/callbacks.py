import os

import s3fs
import torch
from lightning import pytorch as pl


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
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
