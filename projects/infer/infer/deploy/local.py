import logging
import time
from pathlib import Path
from typing import List

from infer.utils import aggregate_results, build_condor_submit, wait

from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor
from utils.logging import configure_logging


def deploy_local(
    ip_address: str,
    image: str,
    model_name: str,
    model_repo_dir: Path,
    shifts: List[float],
    num_shifts: int,
    background_fnames: List[Path],
    injection_set_fname: Path,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    psd_length: float,
    fduration: float,
    output_dir: Path,
    num_parallel_jobs: int,
    model_version: int = -1,
):
    configure_logging(verbose=True)

    job = build_condor_submit(
        ip_address,
        model_name,
        shifts,
        num_shifts,
        background_fnames,
        injection_set_fname,
        inference_sampling_rate,
        ifos,
        batch_size,
        integration_window_length,
        cluster_window_length,
        psd_length,
        fduration,
        output_dir,
        num_parallel_jobs,
        model_version,
    )

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    server_log = log_dir / "server.log"
    logging.info("Starting server")

    with serve(model_repo_dir, image, log_file=server_log, wait=True):
        # launch inference jobs via condor
        logging.info("Server online")
        time.sleep(1)
        monitor = ServerMonitor(
            model_name=model_name,
            ips="localhost",
            filename=log_dir / f"server-stats-{batch_size}.csv",
            model_version=model_version,
            name="monitor",
            rate=10,
        )
        with monitor:
            cluster = job.build_submit(fancyname=False)
            wait(cluster)

    logging.info("Aggregating results")
    aggregate_results(output_dir)
