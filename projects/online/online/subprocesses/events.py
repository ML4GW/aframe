from queue import Queue
import logging
from pathlib import Path

from online.subprocesses.wrapper import subprocess_wrapper
from online.utils.gdb import GdbServer, gracedb_factory


@subprocess_wrapper
def event_creation_subprocess(
    event_queue: Queue,
    server: GdbServer,
    outdir: Path,
    amplfi_queue: Queue,
    pastro_queue: Queue,
):
    gdb = gracedb_factory(server, outdir)

    while True:
        event = event_queue.get()
        logging.debug("Putting event in pastro queue")
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
        logging.debug("Putting graceid in amplfi and pastro queues")
        amplfi_queue.put(graceid)
        pastro_queue.put(graceid)
