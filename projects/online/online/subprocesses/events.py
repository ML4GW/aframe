from queue import Queue
import logging
from pathlib import Path
from queue import Empty

from .wrapper import subprocess_wrapper
from online.utils.gdb import GdbServer, gracedb_factory, authenticate
import time


@subprocess_wrapper
def event_creation_subprocess(
    event_queue: Queue,
    server: GdbServer,
    outdir: Path,
    amplfi_queue: Queue,
    pastro_queue: Queue,
):
    gdb = gracedb_factory(server, outdir)
    last_auth = time.time()
    while True:
        try:
            event = event_queue.get_nowait()
            logging.info("Putting event in pastro queue")
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
            logging.info("Putting graceid in amplfi and pastro queues")
            amplfi_queue.put(graceid)
            pastro_queue.put(graceid)
        except Empty:
            time.sleep(1e-3)
            # Re-authenticate every 1000 seconds so that
            # the scitoken doesn't expire. Doing it in this
            # loop as it's the earliest point of submission
            if last_auth - time.time() > 1000:
                authenticate()
                last_auth = time.time()
