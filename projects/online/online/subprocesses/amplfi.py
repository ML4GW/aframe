import logging
from pathlib import Path
import torch
from typing import Optional
from amplfi.train.data.utils.utils import ParameterSampler
from torch.multiprocessing import Array, Queue
from online.utils.searcher import Event
from online.utils.gdb import GdbServer, gracedb_factory
from online.utils.pe import postprocess_samples
from astropy import io
from .utils import subprocess_wrapper
from online.utils.email_alerts import send_detection_email

logger = logging.getLogger("amplfi-subprocess")


@subprocess_wrapper
def amplfi_subprocess(
    amplfi_queue: Queue,
    server: GdbServer,
    outdir: Path,
    inference_params: list[str],
    amplfi_parameter_sampler: ParameterSampler,
    shared_samples: Array,
    emails: Optional[list[str]] = None,
    email_far_threshold: float = 1e-6,
    nside: int = 32,
):
    logger.info("amplfi subprocess initialized")
    while True:
        arg = amplfi_queue.get()
        gdb = gracedb_factory(server, outdir)
        if isinstance(arg[0], Event):
            event, amplfi_ifos = arg
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples), (-1, len(inference_params))
            )
            logger.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event.gpstime,
                inference_params,
                amplfi_parameter_sampler,
            )

            logger.info("Creating low resolution skymap")
            skymap = result.to_skymap(nside, use_distance=False)
            fits_skymap = io.fits.table_to_hdu(skymap)

            graceid = amplfi_queue.get()

            if emails is not None and event.far < email_far_threshold:
                logger.info("Sending detection email")
                send_detection_email(emails, result, event, graceid, server)

            logger.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(
                result, fits_skymap, graceid, event.event_dir, amplfi_ifos
            )

            logger.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(
                result, graceid, event.event_dir, amplfi_ifos
            )
            logger.info("Submitted all PE")
        else:
            graceid = arg
            event, amplfi_ifos = amplfi_queue.get()
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples), (-1, len(inference_params))
            )
            logger.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event.gpstime,
                inference_params,
                amplfi_parameter_sampler,
            )

            if emails is not None and event.far < email_far_threshold:
                logger.info("Sending detection email")
                send_detection_email(emails, result, event, graceid, server)

            logger.info("Creating low resolution skymap")
            skymap = result.to_skymap(nside, use_distance=False)
            fits_skymap = io.fits.table_to_hdu(skymap)

            logger.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(
                result, fits_skymap, graceid, event.event_dir, amplfi_ifos
            )

            logger.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(
                result, graceid, event.event_dir, amplfi_ifos
            )

            logger.info("Submitting ligo.skymap mollweide plots")
            gdb.submit_skymap_plots(graceid, event.event_dir)
            logger.info("Submitted all PE")
