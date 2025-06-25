import logging
import torch
from typing import Optional, TYPE_CHECKING
from amplfi.train.data.utils.utils import ParameterSampler
from torch.multiprocessing import Array, Queue
from online.utils.searcher import Event
from online.utils.pe import postprocess_samples
from astropy import io
from .utils import subprocess_wrapper
from online.utils.email_alerts import send_detection_email

if TYPE_CHECKING:
    from online.utils.gdb import GraceDb

logger = logging.getLogger("amplfi-subprocess")


@subprocess_wrapper
def amplfi_subprocess(
    amplfi_queue: Queue,
    gdb: "GraceDb",
    inference_params: list[str],
    amplfi_parameter_sampler: ParameterSampler,
    shared_samples: Array,
    emails: Optional[list[str]] = None,
    email_far_threshold: float = 1e-6,
    nside: int = 64,
    min_samples_per_pix: int = 5,
    use_distance: bool = True,
):
    logger.info("amplfi subprocess initialized")
    # override with subprocesses logger
    gdb.logger = logger

    while True:
        arg = amplfi_queue.get()
        if isinstance(arg[0], Event):
            event: Event
            amplfi_ifos: list[str]

            event, amplfi_ifos = arg
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples[:]), (-1, len(inference_params))
            )
            logger.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event.gpstime,
                inference_params,
                amplfi_parameter_sampler,
            )

            logger.info("Creating low resolution skymap")
            skymap = result.to_skymap(
                nside, min_samples_per_pix, use_distance=use_distance
            )
            fits_skymap = io.fits.table_to_hdu(skymap)

            graceid = amplfi_queue.get()

            if emails is not None and event.far < email_far_threshold:
                logger.info("Sending detection email")
                send_detection_email(
                    emails, result, event, graceid, gdb.url(graceid)
                )

            logger.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(
                result,
                fits_skymap,
                graceid,
                event.event_dir,
            )

            logger.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(
                result, graceid, event.event_dir, amplfi_ifos
            )

            logger.info("Submitting ligo.skymap mollweide plots")
            gdb.submit_skymap_plots(graceid, event.event_dir)
            logger.info("Submitted all PE")
        else:
            graceid = arg
            event: Event
            amplfi_ifos: list[str]
            event, amplfi_ifos = amplfi_queue.get()

            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples[:]), (-1, len(inference_params))
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
                send_detection_email(
                    emails, result, event, graceid, gdb.url(graceid)
                )

            logger.info("Creating low resolution skymap")
            skymap = result.to_skymap(nside, use_distance=use_distance)
            fits_skymap = io.fits.table_to_hdu(skymap)

            logger.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(
                result,
                fits_skymap,
                graceid,
                event.event_dir,
            )

            logger.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(
                result, graceid, event.event_dir, amplfi_ifos
            )

            logger.info("Submitting ligo.skymap mollweide plots")
            gdb.submit_skymap_plots(graceid, event.event_dir)
            logger.info("Submitted all PE")
