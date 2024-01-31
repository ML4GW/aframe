import os

from lightning.pytorch.cli import LightningCLI

from train.data import BaseAframeDataset
from train.model import AframeBase
from utils.logging import configure_logging


class AframeCLI(LightningCLI):
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

    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
