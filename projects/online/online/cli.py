import logging
from datetime import datetime, timezone

import jsonargparse

from online.main import main
from utils.logging import configure_logging


def build_parser():
    # use omegaconf to suppor env var interpolation
    parser = jsonargparse.ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")

    parser.link_arguments(
        "inference_params",
        "amplfi_hl_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )

    parser.link_arguments(
        "inference_params",
        "amplfi_hlv_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )

    return parser


def cli(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    # Create a new log file each time we start using the current UTC time
    logdir = args.outdir / "log"
    logdir.mkdir(exist_ok=True, parents=True)
    log_suffix = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    configure_logging(
        args.outdir / "log" / f"deploy_{log_suffix}.log", args.verbose
    )
    # matplotlib and h5py have some debug-level logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    args.pop("config")
    args.pop("verbose")
    args = parser.instantiate_classes(args)

    main(**vars(args))


if __name__ == "__main__":
    cli()
