import logging
from pathlib import Path
import torch
from amplfi.train.data.utils.utils import ParameterSampler
from torch.multiprocessing import Array, Queue
from online.utils.gdb import GdbServer, gracedb_factory
from online.utils.pe import postprocess_samples
from astropy import io
from online.subprocesses.wrapper import subprocess_wrapper


@subprocess_wrapper
def amplfi_subprocess(
    amplfi_queue: Queue,
    server: GdbServer,
    outdir: Path,
    inference_params: list[str],
    amplfi_parameter_sampler: ParameterSampler,
    shared_samples: Array,
    nside: int = 32,
):
    logger = logging.getLogger("amplfi-subprocess")
    gdb = gracedb_factory(server, outdir)
    logger.info("amplfi subprocess initialized")
    while True:
        arg = amplfi_queue.get()
        if isinstance(arg, float):
            event_time = arg
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples), (-1, len(inference_params))
            )
            logger.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event_time,
                inference_params,
                amplfi_parameter_sampler,
            )

            logger.info("Creating low resolution skymap")
            skymap = result.to_skymap(nside, use_distance=False)
            fits_skymap = io.fits.table_to_hdu(skymap)

            graceid = amplfi_queue.get()

            logger.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(result, fits_skymap, graceid, event_time)

            logger.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(result, graceid, event_time)
            logger.info("Submitted all PE")
        else:
            graceid = arg
            event_time = amplfi_queue.get()
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples), (-1, len(inference_params))
            )
            logger.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event_time,
                inference_params,
                amplfi_parameter_sampler,
            )

            logger.info("Creating low resolution skymap")
            skymap = result.to_skymap(nside, use_distance=False)
            fits_skymap = io.fits.table_to_hdu(skymap)

            logger.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(result, fits_skymap, graceid, event_time)

            logger.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(result, graceid, event_time)
            logger.info("Submitted all PE")
