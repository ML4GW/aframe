import logging
from pathlib import Path

from torch.multiprocessing import Queue

from online.utils.gdb import GdbServer, gracedb_factory
from online.utils.pastro import fit_or_load_pastro


from .wrapper import subprocess_wrapper


@subprocess_wrapper
def pastro_subprocess(
    pastro_queue: Queue,
    background_path,
    foreground_path,
    rejected_path,
    astro_event_rate,
    server: GdbServer,
    outdir: Path,
):
    logger = logging.getLogger("pastro-process")
    logger.info("pastro subprocess initialized")
    pastro_model = fit_or_load_pastro(
        outdir / "pastro.pkl",
        background_path,
        foreground_path,
        rejected_path,
        astro_event_rate=astro_event_rate,
    )
    gdb = gracedb_factory(server, outdir)
    while True:
        event = pastro_queue.get()
        logger.info("Calculating p_astro")
        pastro = pastro_model(event.detection_statistic)
        graceid = pastro_queue.get()

        logger.info(f"Submitting p_astro: {pastro}")
        gdb.submit_pastro(float(pastro), graceid, event.gpstime)
        logger.info("Submitted p_astro")
