from .make_event_pages import main as make_event_pages
from .make_summary_page import main as make_summary_page

import argparse
import time
from pathlib import Path

from utils.logging import configure_logging


def cli():
    configure_logging()

    parser = argparse.ArgumentParser(description="Process event outputs")
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

    if not args.outdir.exists():
        args.outdir.mkdir(exist_ok=True, parents=True)

    event_dir = args.run_dir / "output" / "events"
    while True:
        make_event_pages(event_dir, args.outdir)
        make_summary_page(args.run_dir, args.outdir)
        time.sleep(60)


if __name__ == "__main__":
    cli()
