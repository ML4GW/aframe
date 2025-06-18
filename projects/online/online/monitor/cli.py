from .make_event_pages import main as make_event_pages
from .make_summary_page import main as make_summary_page

import time
from pathlib import Path

import yaml
from jsonargparse import ArgumentParser

from utils.logging import configure_logging


def cli():
    configure_logging()

    parser = ArgumentParser(description="Process event outputs")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Root directory of the online search",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    # Load in online configuration to get various values
    # for making plots; e.g., sample_rate, event_position
    config = args.run_dir / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")
    with open(config, "r") as f:
        online_args = yaml.safe_load(f)

    if not args.outdir.exists():
        args.outdir.mkdir(exist_ok=True, parents=True)

    while True:
        make_event_pages(**vars(args), online_args=online_args)
        make_summary_page(**vars(args), online_args=online_args)
        time.sleep(60)


if __name__ == "__main__":
    cli()
