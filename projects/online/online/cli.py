from datetime import datetime, timezone

import jsonargparse
from online.main import main

from utils.logging import configure_logging


def build_parser():
    # use omegaconf to suppor env var interpolation
    parser = jsonargparse.ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--verbose", type=str, default=False)
    parser.add_function_arguments(main)
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
    main(**vars(args))


if __name__ == "__main__":
    cli()
