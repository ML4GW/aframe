from pathlib import Path
import logging
import time

from .utils.parse_logs import estimate_tb, pipeline_online
from .pages import EventPage, SummaryPage

from gwpy.time import tconvert
from datetime import datetime, timezone

logging.getLogger("aframe-monitor")


def main(
    run_dir: Path,
    out_dir: Path,
    online_args: dict,
    start_time: float = None,
    update_cadence: int = 10,
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

    if start_time is None:
        start_time = float(tconvert(datetime.now(timezone.utc)))

    # Estimate analysis live time since the given start time
    tb = estimate_tb(run_dir, start_time)
    logging.info(f"Estimated analysis live time: {tb:.2f} seconds")

    previous_update_time = time.time()

    summary_page = SummaryPage(start_time, run_dir, out_dir)
    while True:
        detected_events = [
            event
            for event in detected_event_dir.iterdir()
            if float(event.name.split("_")[1]) >= start_time
        ]

        # The event page will be created/updated only if the event directory
        # is missing expected plots. Otherwise it will be skipped.
        for event in detected_events:
            event_page = EventPage(event, online_args, run_dir, out_dir)
            event_page.create()

        # The dataframe file will not exist until the first event is processed
        if summary_page.dataframe_file.exists():
            logging.info("Updating summary page")
            update_time = time.time()
            # If the pipeline is running, update `tb` with the time
            # since the last update. If not, set the previous update time
            # to the current time so that inactive time is not counted
            # down the line. This isn't super precise, but it shouldn't be
            # off by more than `update_cadence` seconds.
            if pipeline_online():
                tb += update_time - previous_update_time
            previous_update_time = update_time
            summary_page.create(tb)
        else:
            logging.info("Skipping summary page update")

        # Sleep for a minute before checking for new events again
        time.sleep(update_cadence)
