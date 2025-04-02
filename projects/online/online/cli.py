import logging

import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import jsonargparse

from online.main import main


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


def configure_logging(logdir: Path, verbose: bool = False):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger()

    # set logging to use UTC time
    logging.Formatter.converter = lambda *args: datetime.now(
        tz=timezone.utc
    ).timetuple()

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_log_dir = logdir / timestamp
    run_log_dir.mkdir(exist_ok=True, parents=True)

    # set up the timed rotating file handler
    formatter = logging.Formatter(log_format)

    # ensure formatter also uses UTC time
    formatter.converter = lambda *args: datetime.now(
        tz=timezone.utc
    ).timetuple()

    log_file = run_log_dir / "online.log"

    # create a timed rotating file handler
    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        backupCount=0,
        utc=True,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Logging initialized in directory: {run_log_dir}")

    # matplotlib and h5py have some debug-level logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)


def cli(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    # create a new log directory each time we start using the current UTC time
    logdir = args.outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir, args.verbose)

    args.pop("config")
    args.pop("verbose")
    args = parser.instantiate_classes(args)

    main(**vars(args))


if __name__ == "__main__":
    cli()
