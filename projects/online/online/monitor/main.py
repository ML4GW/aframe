from pathlib import Path
from .make_event_page import main as make_event_page
from .make_summary_page import main as make_summary_page

from .process_events import process_events
from gwpy.time import tconvert
from datetime import datetime, timezone

import logging
import time


def main(
    run_dir: Path, outdir: Path, online_args: dict, start_time: float = None
):
    """
    Main function to monitor the online search for new events and process them.
    Args:
        run_dir: Root directory of the online search.
        outdir: Output directory for processed data.
        online_args: Configuration parameters for the online search.
        start_time:
            The earliest GPS time to consider for processing events.
            If None, the current GPS time is used.

    """
    detected_event_dir = run_dir / "output" / "events"
    output_event_dir = outdir / "events"
    if not output_event_dir.exists():
        output_event_dir.mkdir(exist_ok=True, parents=True)

    if start_time is None:
        start_time = float(tconvert(datetime.now(timezone.utc)))

    df = None
    while True:
        detected_events = {
            event.name
            for event in detected_event_dir.iterdir()
            if float(event.name.split("_")[1]) >= start_time
        }
        processed_events = {event.name for event in output_event_dir.iterdir()}
        new_events = detected_events.difference(processed_events)
        new_events = [detected_event_dir / event for event in new_events]
        if new_events:
            logging.info(f"Processing {len(new_events)} new events")
            events = sorted(new_events)
            df = process_events(events, output_event_dir, online_args)
            for event, url in zip(events, df["url"]):
                make_event_page(event, url, output_event_dir)
        else:
            logging.info("No new events detected")

        logging.info("Updating summary page")
        make_summary_page(run_dir, outdir, start_time, df)

        # Sleep for a minute before checking for new events again
        time.sleep(60)
