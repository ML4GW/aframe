from .main import main

import logging
from pathlib import Path

import yaml
from jsonargparse import ArgumentParser

from utils.logging import configure_logging

logger = logging.getLogger("aframe-monitor")


def cli():
    parser = ArgumentParser(description="Process event outputs")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Root directory of the online search",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Earliest GPS time to consider for processing events.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default="monitor.log",
        help="Log file name for the monitoring process",
    )
    parser.add_argument(
        "--update_cadence",
        type=float,
        default=60,
        help="Update cadence for the monitoring process in seconds",
    )
    args = parser.parse_args()

    if not args.out_dir.exists():
        args.out_dir.mkdir(exist_ok=True, parents=True)

    configure_logging(filename=args.out_dir / args.log_file)
    args.pop("log_file")

    # Load in online configuration to get various values
    # for making plots; e.g., sample_rate, event_position
    config = args.run_dir / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")
    with open(config, "r") as f:
        online_args = yaml.safe_load(f)

    main(**vars(args), online_args=online_args)


if __name__ == "__main__":
    cli()
