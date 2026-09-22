import logging
import time
from pathlib import Path

from .pages import EventPage, SummaryPage
from .utils.segments import load_segments


def find_events(event_dir: Path, start_time: float) -> list[Path]:
    """
    Find the event directories written since `start_time`. The
    directory doesn't exist until the search detects something.
    """
    if not event_dir.exists():
        return []

    events = [
        event
        for event in event_dir.iterdir()
        if float(event.name.split("_")[1]) > start_time
    ]
    return sorted(events)


def update(
    run_dir: Path,
    out_dir: Path,
    online_args: dict,
    start_time: float | None,
    summary_page: SummaryPage,
    logger: logging.Logger,
) -> None:
    """Bring the event pages and the summary page up to date"""
    for event in find_events(
        run_dir / "output" / "events", summary_page.start_time
    ):
        EventPage(event, online_args, run_dir, out_dir, logger).create()

    segments = load_segments(run_dir, start_time)
    logger.info("Updating summary page")
    summary_page.create(segments)


def main(
    run_dir: Path,
    out_dir: Path,
    online_args: dict,
    start_time: float = None,
    update_cadence: int = 10,
    logger: logging.Logger = None,
):
    """
    Main function to monitor the online search for new events and process them.
    Args:
        run_dir: Root directory of the online search.
        outdir: Output directory for processed data.
        online_args: Configuration parameters for the online search.
        start_time:
            The earliest GPS time to consider for processing events.
            If None and an event dataframe file exists, the oldest
            event time will be used. If the file does not exist,
            the current GPS time is used.
        update_cadence:
            The interval in seconds to check for new events.
        logger: Logger object for standardizing logging output

    """
    if logger is None:
        logger = logging.getLogger()

    summary_page = SummaryPage(start_time, run_dir, out_dir, logger)
    while True:
        update(run_dir, out_dir, online_args, start_time, summary_page, logger)
        time.sleep(update_cadence)
