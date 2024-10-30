from lightning.pytorch.cli import LightningCLI

from train.callbacks import WandbSaveConfig
from train.data import BaseAframeDataset
from train.model import AframeBase


class AframeCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        # hack into init to hardcode
        # the WandbSaveConfig callback
        kwargs["save_config_callback"] = WandbSaveConfig
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        # link data arguments to model;
        # some models require information about
        # sequence length before hand, so we need
        # to link sample rate and kernel length
        parser.link_arguments(
            "data.num_ifos",
            "model.init_args.arch.init_args.num_ifos",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.init_args.kernel_length",
            "model.init_args.arch.init_args.kernel_length",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.init_args.sample_rate",
            "model.init_args.arch.init_args.sample_rate",
            apply_on="parse",
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
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
    return cli


if __name__ == "__main__":
    main()
