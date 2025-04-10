from queue import Queue
import logging
from pathlib import Path

from .utils import subprocess_wrapper
from online.utils.gdb import GdbServer, gracedb_factory

logger = logging.getLogger("event-creation-subprocess")


@subprocess_wrapper
def event_creation_subprocess(
    event_queue: Queue,
    server: GdbServer,
    outdir: Path,
    amplfi_queue: Queue,
    pastro_queue: Queue,
):
    logger.info("event creation subprocess initialized")

    while True:
        event = event_queue.get()
        gdb = gracedb_factory(server, outdir)
        logger.debug("Putting event in pastro queue")
        pastro_queue.put(event)

        # write event information to disk
        # and submit it to gracedb
        event.write(outdir)
        response = gdb.submit(event)
        # Get the event's graceid for submitting
        # further data products
        if server == "local":
            # The local gracedb client just returns the filename
            graceid = response
        else:
            graceid = response.json()["graceid"]
        logger.debug("Putting graceid in amplfi and pastro queues")
        amplfi_queue.put(graceid)
        pastro_queue.put(graceid)
