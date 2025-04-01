import logging
from pathlib import Path
import torch
from amplfi.train.data.utils.utils import ParameterSampler
from torch.multiprocessing import Array, Queue
from online.utils.gdb import GdbServer, gracedb_factory
from online.utils.pe import (
    create_histogram_skymap,
    postprocess_samples,
)

from .wrapper import subprocess_wrapper


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
    gdb = gracedb_factory(server, outdir)

    while True:
        arg = amplfi_queue.get()
        if isinstance(arg, float):
            event_time = arg
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples), (-1, len(inference_params))
            )
            logging.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event_time,
                inference_params,
                amplfi_parameter_sampler,
            )

            logging.info("Creating low resolution skymap")
            skymap, mollview_map = create_histogram_skymap(
                result.posterior["ra"], result.posterior["dec"], nside
            )
            graceid = amplfi_queue.get()

            logging.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(
                result, mollview_map, skymap, graceid, event_time
            )

            logging.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(result, graceid, event_time)
            logging.info("Submitted all PE")
        else:
            graceid = arg
            event_time = amplfi_queue.get()
            descaled_samples = torch.reshape(
                torch.Tensor(shared_samples), (-1, len(inference_params))
            )
            logging.info("Post-processing samples")
            result = postprocess_samples(
                descaled_samples,
                event_time,
                inference_params,
                amplfi_parameter_sampler,
            )

            logging.info("Creating low resolution skymap")
            skymap, mollview_map = create_histogram_skymap(
                result.posterior["ra"], result.posterior["dec"], nside
            )

            logging.info("Submitting posterior and low resolution skymap")
            gdb.submit_low_latency_pe(
                result, mollview_map, skymap, graceid, event_time
            )

            logging.info("Launching ligo-skymap-from-samples")
            gdb.submit_ligo_skymap_from_samples(result, graceid, event_time)
            logging.info("Submitted all PE")
