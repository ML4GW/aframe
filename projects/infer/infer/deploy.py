import logging
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import List

from ledger.events import EventSet, RecoveredInjectionSet

from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor
from pycondor.cluster import JobStatus
from pycondor.job import Job


def aggregate_results(output_directory: Path):
    """
    Combine results from across segments into a single
    background file and foreground file. Remove the directory
    containing the individual segment results.
    """
    background, foreground = EventSet(), RecoveredInjectionSet()
    for data_dir in (output_directory / "tmp").iterdir():
        bckground = EventSet.read(data_dir / "background.hdf5")
        frground = RecoveredInjectionSet.read(data_dir / "foreground.hdf5")

        background.append(bckground)
        foreground.append(frground)

    background.write(output_directory / "background.hdf5")
    foreground.write(output_directory / "foreground.hdf5")
    shutil.rmtree(output_directory / "tmp")


def deploy(
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
    param_names = "background_fname,shift"
    parameters = ""
    for fname in background_fnames:
        for i in range(num_shifts):
            _shifts = "[" + ",".join([str(s * i) for s in shifts]) + "]"
            parameters += f"{fname},{_shifts}\n"

    condor_dir = output_dir / "condor"
    condor_dir.mkdir(parents=True, exist_ok=True)
    param_file = condor_dir / "params.txt"
    with open(param_file, "w") as f:
        f.write(parameters)

    log_pattern = "infer-$(ProcID).log"
    output_pattern = "tmp/output-$(ProcID)"

    arguments = f"""
    --client.address={ip_address}:8001
    --client.model_name {model_name}
    --client.model_version {model_version}
    --data.ifos=[{','.join(ifos)}]
    --data.batch_size {batch_size}
    --data.inference_sampling_rate {inference_sampling_rate}
    --data.injection_set_fname {injection_set_fname}
    --postprocessor.integration_window_length {integration_window_length}
    --postprocessor.cluster_window_length {cluster_window_length}
    --postprocessor.psd_length {psd_length}
    --postprocessor.fduration {fduration}
    --data.background_fname $(background_fname)
    --data.shifts=$(shift)
    --outdir {output_dir / output_pattern}
    --logfile {output_dir / "log" / log_pattern}
    """
    arguments = dedent(arguments).replace("\n", " ")
    log_dir = condor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = f"{param_names} from {param_file}"

    job = Job(
        name="infer-clients",
        executable=shutil.which("infer"),
        error=str(log_dir),
        log=str(log_dir),
        output=str(log_dir),
        submit=str(condor_dir),
        request_memory="6G",
        request_disk="1G",
        request_cpus="1",
        queue=queue,
        arguments=arguments,
        extra_lines=[f"max_materialize = {num_parallel_jobs}"],
    )
    server_log = condor_dir / "server.log"
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
            while not cluster.check_status(JobStatus.COMPLETED, how="all"):
                if cluster.check_status(
                    [JobStatus.FAILED, JobStatus.CANCELLED], how="any"
                ):
                    for proc in cluster.procs:
                        print(proc.err)
                    cluster.rm()
                    raise ValueError("Something went wrong!")

    aggregate_results(output_dir)
