from queue import Queue
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import subprocess_wrapper

if TYPE_CHECKING:
    from ligo.gracedb.rest import GraceDb

logger = logging.getLogger("event-creation-subprocess")


@subprocess_wrapper
def event_creation_subprocess(
    event_queue: Queue,
    gdb: "GraceDb",
    outdir: Path,
    amplfi_queue: Queue,
    pastro_queue: Queue,
):
    logger.info("event creation subprocess initialized")

    while True:
        event = event_queue.get()
        logger.debug("Putting event in pastro queue")
        pastro_queue.put(event)

        # write event information to disk
        # and submit it to gracedb
        event.write(outdir)
        response = gdb.submit(event)
        # Get the event's graceid for submitting
        # further data products
        if gdb.server == "local":
            # The local gracedb client just returns the filename
            graceid = response
        else:
            graceid = response.json()["graceid"]
        logger.debug("Putting graceid in amplfi and pastro queues")
        amplfi_queue.put(graceid)
        pastro_queue.put(graceid)
