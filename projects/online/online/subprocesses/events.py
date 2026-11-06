from queue import Queue
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from online.utils.email_alerts import send_submission_failure_email
from .utils import subprocess_wrapper

if TYPE_CHECKING:
    from online.utils.gdb import GraceDb

logger = logging.getLogger("event-creation-subprocess")


@subprocess_wrapper
def event_creation_subprocess(
    event_queue: Queue,
    gdb: "GraceDb",
    outdir: Path,
    amplfi_queue: Queue,
    pastro_queue: Queue,
    emails: list[str] | None = None,
):
    logger.info("event creation subprocess initialized")

    # override with subprocesses logger
    gdb.logger = logger

    # Need to set up the producer withon the subprocess that uses it.
    # In ligo-gracedb 2.16, an error raises if the broker can't be reached,
    # so keep the search running and let the `submit` function try to create
    # one again if it fails here.
    try:
        gdb.setup_kafka_producer()
    except Exception as e:
        logger.error(f"Couldn't set up Kafka producer: {e}")
        if emails is not None:
            send_submission_failure_email(
                f"Couldn't set up Kafka producer: {e}",
                traceback.format_exc(),
                emails,
            )

    while True:
        event = event_queue.get()
        logger.debug("Putting event in pastro queue")
        pastro_queue.put(event)

        # write event information to disk and submit it to gracedb.
        # `submit` returns None when the event may not have been created
        event.write(outdir)
        graceid, error, tb = None, None, ""
        try:
            graceid = gdb.submit(event)
        except Exception as e:
            error, tb = str(e), traceback.format_exc()
        else:
            if graceid is None:
                error = f"{event} may not exist in GraceDB; see the logs"

        if error is not None:
            logger.error(f"Event submission failed: {error}")
            if emails is not None:
                send_submission_failure_email(error, tb, emails)

        logger.debug(f"Putting graceid {graceid} in amplfi and pastro queues")
        amplfi_queue.put(graceid)
        pastro_queue.put(graceid)
