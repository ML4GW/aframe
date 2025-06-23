from typing import TYPE_CHECKING
import logging
from logging.handlers import TimedRotatingFileHandler, QueueHandler
from datetime import datetime, timezone
import sys
from multiprocessing import Process, Queue
from online.subprocesses.utils import subprocess_wrapper

if TYPE_CHECKING:
    from multiprocessing import Queue
    from pathlib import Path


# custom formatter that adds processing time
class TimestampFormatter(logging.Formatter):
    def format(self, record):
        # original log timestamp using UTC converter
        recorded = datetime.fromtimestamp(record.created, tz=timezone.utc)
        recorded = recorded.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        # time log is processed by logging subprocess
        processed = datetime.now(tz=timezone.utc)
        processed = processed.strftime("%H:%M:%S,%f")[:-3]

        message = (
            f"{recorded} ({processed}) - {record.name} "
            f"- {record.levelname} - {record.getMessage()}"
        )
        return message


def configure_logging(logdir: "Path", verbose: bool = False):
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger()

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_log_dir = logdir / timestamp
    run_log_dir.mkdir(exist_ok=True, parents=True)

    # set up the timed rotating file handler
    formatter = TimestampFormatter()

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

    # matplotlib and h5py have some debug-level
    # logging we want to suppress
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)


@subprocess_wrapper
def logging_subprocess(queue: Queue, log_dir: "Path", verbose: bool = False):
    """
    Logging subprocess that will be started in a separate Process.
    Ingests and processes logs from a `Queue` that is populated via
    many separate subprocesses
    """
    configure_logging(log_dir, verbose)
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def setup_logging(
    log_dir: "Path", error_queue: Queue, level: bool = True
) -> tuple[Queue, Process]:
    """
    Called in the main thread to setup a logging queue and
    initiate the logging subprocess
    """

    # create a new base log directory each time
    # we start using the current UTC time
    log_dir.mkdir(exist_ok=True, parents=True)

    # queue which will ingest logging records
    # from subprocesses, and pass them to main
    # logger subprocess
    log_queue = Queue()
    args = (error_queue, level, "logging", None, log_queue, log_dir)
    listener = Process(target=logging_subprocess, args=args)
    listener.start()

    h = QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(level)

    return log_queue, listener
