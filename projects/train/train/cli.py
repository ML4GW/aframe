import os

import torch
from lightning.pytorch.cli import LightningCLI

from train.callbacks import WandbSaveConfig
from train.data import BaseAframeDataset
from train.model import AframeBase
from utils.logging import configure_logging


class AframeCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        # hack into init to hardcode
        # parser_mode to omegaconf for all subclasses

        kwargs["save_config_callback"] = WandbSaveConfig
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.num_ifos",
            "model.init_args.arch.init_args.num_ifos",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.init_args.valid_stride",
            "model.init_args.metric.init_args.stride",
        )
        parser.add_argument(
            "--matmul_precision",
            type=str,
            default="highest",
        )
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
        )


def main(args=None):
    cli = AframeCLI(
        AframeBase,
        BaseAframeDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
    # CSV Logger and WandB logger use different
    # names for this variable. Unfortunate.
    log_dir = cli.trainer.logger.log_dir or cli.trainer.logger.save_dir
    if not log_dir.startswith("s3://"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")
        configure_logging(log_file, verbose=True)
    else:
        configure_logging()

    torch.set_float32_matmul_precision(cli.config["matmul_precision"])
    cli.trainer.fit(
        cli.model, cli.datamodule, ckpt_path=cli.config["ckpt_path"]
    )


if __name__ == "__main__":
    main()
