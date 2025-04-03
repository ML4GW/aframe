import logging
from pathlib import Path

from torch.multiprocessing import Queue
from online.utils.gdb import GdbServer, gracedb_factory
from online.subprocesses.wrapper import subprocess_wrapper

logger = logging.getLogger("ligo-skymap-subprocess")


@subprocess_wrapper
def ligo_skymap_subprocess(
    ligo_skymap_queue: Queue,
    server: GdbServer,
    outdir: Path,
):
    gdb = gracedb_factory(server, outdir)
    logger.info("ligo-skymap subprocess initialized")
    while True:
        result, graceid, event_time = ligo_skymap_queue.get()
        logger.info("Launching ligo-skymap-from-samples")
        gdb.submit_ligo_skymap_from_samples(result, graceid, event_time)
        logger.info(f"Submitted all PE for event at {event_time}")
