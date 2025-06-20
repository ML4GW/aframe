from pathlib import Path
import logging
import time

from .make_summary_page import main as make_summary_page
from .process_events import process_events
from .parse_logs import estimate_tb, get_pipeline_status

from gwpy.time import tconvert
from datetime import datetime, timezone


def main(
    run_dir: Path,
    outdir: Path,
    online_args: dict,
    start_time: float = None,
    update_cadence: int = 60,
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
        update_cadence:
            The interval in seconds to check for new events.

    """
    detected_event_dir = run_dir / "output" / "events"
    output_event_dir = outdir / "events"
    if not output_event_dir.exists():
        output_event_dir.mkdir(exist_ok=True, parents=True)

    if start_time is None:
        start_time = float(tconvert(datetime.now(timezone.utc)))

    # Estimate analysis live time since the given start time
    tb = estimate_tb(run_dir, start_time)
    logging.info(f"Estimated analysis live time: {tb:.2f} seconds")

    previous_update_time = time.time()

    while True:
        # Check for new events that haven't been processed
        detected_events = {
            event.name
            for event in detected_event_dir.iterdir()
            if float(event.name.split("_")[1]) >= start_time
        }
        processed_events = {event.name for event in output_event_dir.iterdir()}
        new_events = [
            detected_event_dir / event
            for event in detected_events - processed_events
        ]

        df = process_events(sorted(new_events), output_event_dir, online_args)

        # df will be None only if there are no previously processed events
        if df is not None:
            logging.info("Updating summary page")
            update_time = time.time()
            # If the pipeline is running, update `tb` with the time
            # since the last update. If not, set the previous update time
            # to the current time so that inactive time is not counted
            # down the line. This isn't super precise, but it shouldn't be
            # off by more than `update_cadence` seconds.
            if get_pipeline_status():
                tb += update_time - previous_update_time
            previous_update_time = update_time
            make_summary_page(run_dir, outdir, start_time, df, tb)
        else:
            logging.info("Skipping summary page update")

        # Sleep for a minute before checking for new events again
        time.sleep(update_cadence)
