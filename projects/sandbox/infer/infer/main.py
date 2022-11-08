import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from infer.utils import SequenceManager
from typeo import scriptify

from bbhnet.logging import configure_logging
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor

# turn off debugging messages from request libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@scriptify
def main(
    model_repo_dir: str,
    model_name: str,
    data_dir: Path,
    write_dir: Path,
    fields: Iterable[str],
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    num_workers: int,
    throughput_per_gpu: float,
    streams_per_gpu: int,
    model_version: int = -1,
    max_seconds: float = 131072,
    base_sequence_id: int = 1001,
    log_file: Optional[Path] = None,
    fduration: Optional[float] = None,
    verbose: bool = False,
):
    configure_logging(log_file, verbose)
    if log_file is not None:
        server_log_file = log_file.parent / "server.log"
    else:
        server_log_file = None

    gpus = os.getenv("CUDA_VISIBLE_DEVICES")
    num_gpus = len(gpus.split(",")) if gpus is not None else 1

    # spin up a triton server and don't move on until it's ready
    with serve(model_repo_dir, wait=True, log_file=server_log_file):
        client = InferenceClient("localhost:8001", model_name, model_version)
        manager = SequenceManager(
            data_dir,
            write_dir,
            fields,
            client,
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            batch_size=batch_size,
            fduration=fduration,
            throughput=throughput_per_gpu * num_gpus,
            num_io_workers=num_workers,
            max_streams=streams_per_gpu * num_gpus,
            max_seconds=max_seconds,
            base_sequence_id=base_sequence_id,
        )

        # create a monitor which will recored per-model
        # inference stats for profiling purposes
        monitor = ServerMonitor(
            model_name=model_name,
            ips="localhost",
            filename=log_file.parent / "server-stats.csv",
            model_version=model_version,
            name="monitor",
            rate=10,
        )

        # enter all of these objects as contexts
        with client, manager, monitor:
            while not manager.is_done():
                # check to see if any sequences have finished inference
                # and have been submitted for writing
                future = client.get()
                if future is None:
                    time.sleep(1e-2)
                    continue

                # if they have, wait until writing finishes and
                # log the filename that got written
                while not future.done():
                    time.sleep(1e-2)
                fname = future.result()
                logging.info(f"Wrote inferred segment to file {fname}")


if __name__ == "__main__":
    main()
