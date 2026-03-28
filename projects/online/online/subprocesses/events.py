from queue import Queue
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import certifi
from ligo.gracedb.kafka import GraceDbKafkaProducer

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
):
    logger.info("event creation subprocess initialized")

    # override with subprocesses logger
    gdb.logger = logger

    # Need to create the producer within the subprocess that uses it
    gdb.kafka_producer = GraceDbKafkaProducer(
        bootstrap_servers="kafka-dev.ligo.org:9092",
        service_url=gdb.server.service_url,
        ca_cert_path=certifi.where(),
    )
    while True:
        event = event_queue.get()
        logger.debug("Putting event in pastro queue")
        pastro_queue.put(event)

        # write event information to disk
        # and submit it to gracedb
        event.write(outdir)
        graceid = gdb.submit(event)

        logger.debug(f"Putting graceid {graceid} in amplfi and pastro queues")
        amplfi_queue.put(graceid)
        pastro_queue.put(graceid)
