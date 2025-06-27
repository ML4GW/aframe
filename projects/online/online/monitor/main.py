import logging
import time
from pathlib import Path

from .utils.parse_logs import estimate_tb, pipeline_online
from .pages import EventPage, SummaryPage


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

    detected_event_dir = run_dir / "output" / "events"

    summary_page = SummaryPage(start_time, run_dir, out_dir, logger)
    # Estimate analysis live time since the given start time
    tb = estimate_tb(run_dir, summary_page.start_time)
    logger.info(f"Estimated analysis live time: {tb:.2f} seconds")

    previous_update_time = time.time()

    while True:
        detected_events = [
            event
            for event in detected_event_dir.iterdir()
            if float(event.name.split("_")[1]) >= summary_page.start_time
        ]

        # The event page will be created/updated only if the event directory
        # is missing expected plots. Otherwise it will be skipped.
        for event in detected_events:
            event_page = EventPage(
                event, online_args, run_dir, out_dir, logger
            )
            event_page.create()

        # The dataframe file will not exist until the first event is processed
        if summary_page.dataframe_file.exists():
            logger.info("Updating summary page")
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
            logger.info("Skipping summary page update")

        # Sleep for a minute before checking for new events again
        time.sleep(update_cadence)
