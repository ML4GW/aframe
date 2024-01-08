from pathlib import Path
from typing import List

from infer.utils import aggregate_results, build_condor_submit, wait


def deploy_remote(
    ip_address: str,
    model_name: str,
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
    cluster = job.build_submit(fancyname=False)
    wait(cluster)

    aggregate_results(output_dir)
