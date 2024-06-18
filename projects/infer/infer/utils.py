import logging
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import h5py
import numpy as np
from ledger.events import EventSet, RecoveredInjectionSet

from pycondor.cluster import JobStatus
from pycondor.job import Job
from utils.data import is_analyzeable_segment


def build_condor_submit(
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
    rate: Optional[float] = None,
    model_version: int = -1,
    zero_lag: bool = False,
) -> Job:
    """
    Build a condor submit file that will launch multiple infer jobs in parallel
    """
    param_names = "background_fname,shift"
    parameters = ""

    for fname in background_fnames:
        start, duration = map(float, Path(fname).stem.split("-")[-2:])
        stop = start + duration
        for i in range(num_shifts):
            _shifts = [s * (i + 1) for s in shifts]
            # check if segment is long enough to be analyzed
            if is_analyzeable_segment(start, stop, _shifts, psd_length):
                parameters += f"{fname},'{_shifts}'\n"

        # if its somehow not analyzeable for 0lag then segment
        # length has been set incorrectly, but put this check here anyway
        if zero_lag and is_analyzeable_segment(
            start, stop, [0] * len(shifts), psd_length
        ):
            _shifts = [0 for s in shifts]
            parameters += f"{fname},'{_shifts}'\n"

    condor_dir = output_dir / "condor"
    condor_dir.mkdir(parents=True, exist_ok=True)
    param_file = condor_dir / "params.txt"
    with open(param_file, "w") as f:
        f.write(parameters)

    log_pattern = "infer-$(ProcID).log"
    output_pattern = "tmp/output-$(ProcID)"
    rate = rate or "null"

    arguments = f"""
    --client.address={ip_address}:8001
    --client.model_name {model_name}
    --client.model_version {model_version}
    --data.ifos=[{','.join(ifos)}]
    --data.batch_size {batch_size}
    --data.inference_sampling_rate {inference_sampling_rate}
    --data.injection_set_fname {injection_set_fname}
    --data.rate {rate}
    --postprocessor.integration_window_length {integration_window_length}
    --postprocessor.cluster_window_length {cluster_window_length}
    --postprocessor.psd_length {psd_length}
    --postprocessor.fduration {fduration}
    --data.background_fname $(background_fname)
    --data.shifts=$(shift)
    --outdir {output_dir / output_pattern}
    --logfile {output_dir / "logs" / log_pattern}
    """

    arguments = dedent(arguments).replace("\n", " ")
    log_dir = condor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = f"{param_names} from {param_file}"

    job = Job(
        name="infer-clients",
        executable=shutil.which("infer"),
        error=str(log_dir / "error"),
        log=str(log_dir / "log"),
        output=str(log_dir / "output"),
        suffix="-$(ProcID)",
        submit=str(condor_dir),
        request_memory="6G",
        request_disk="1G",
        request_cpus="1",
        queue=queue,
        arguments=arguments,
        extra_lines=[f"max_materialize = {num_parallel_jobs}"],
    )
    return job


def wait(cluster, sleep: int = 1):
    while not cluster.check_status(JobStatus.COMPLETED, how="all"):
        time.sleep(sleep)
        if cluster.check_status(
            [JobStatus.FAILED, JobStatus.CANCELLED], how="any"
        ):
            for proc in cluster.procs:
                logging.error(proc.err)
            cluster.rm()
            raise RuntimeError("Something went wrong!")


def get_shifts(files: List[Path]):
    shifts = []
    for f in files:
        with h5py.File(f) as f:
            shift = f["parameters"]["shift"][0]
            shifts.append(shift)
    return shifts


def aggregate_results(
    output_directory: Path, ifos: list[str], clean: bool = False
):
    """
    Combine results from across segments into a single
    background file and foreground file. Remove the directory
    containing the individual segment results.
    """
    tmpdir = output_directory / "tmp"

    back_files = np.array([d / "background.hdf5" for d in tmpdir.iterdir()])
    fore_files = [d / "foreground.hdf5" for d in tmpdir.iterdir()]

    # separate 0lag and background events into different files
    shifts = get_shifts(back_files)
    zero_lag = np.array([all(shift == [0] * len(ifos)) for shift in shifts])

    zero_lag_files = back_files[zero_lag]
    back_files = back_files[~zero_lag]

    EventSet.aggregate(
        back_files, output_directory / "background.hdf5", clean=clean
    )
    RecoveredInjectionSet.aggregate(
        fore_files, output_directory / "foreground.hdf5", clean=clean
    )
    if len(zero_lag_files) > 0:
        EventSet.aggregate(
            zero_lag_files, output_directory / "0lag.hdf5", clean=clean
        )
