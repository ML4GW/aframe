import logging
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import List

import h5py
import numpy as np
import psutil
from typeo import scriptify

from aframe.analysis.ledger.events import TimeSlideEventSet
from aframe.deploy import condor
from aframe.logging import configure_logging
from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor

re_fname = re.compile(r"([0-9]{10})-([1-9][0-9]*)\.")
logging.getLogger("urllib3").setLevel(logging.WARNING)


def aggregate_results(output_directory: Path, time_var: float = 1):
    results = {k: defaultdict(list) for k in ["background", "foreground"]}
    for data_dir in (output_directory / "tmp").iterdir():
        for key, value in results.items():
            events = TimeSlideEventSet.read(data_dir / f"{key}.hdf")
            value["time"].append(events.time)
            value["stat"].append(events.detection_statistic)

    for key, value in results.items():
        with h5py.File(output_directory / f"{key}.hdf", "w") as f:
            for k, v in value.items():
                v = np.concatenate(v)
                f[k] = v
            f["var"] = time_var * np.ones_like(v)
    shutil.rmtree(output_directory / "tmp")


def get_ip_address() -> str:
    nets = psutil.net_if_addrs()
    return nets["enp1s0f0"][0].address


@scriptify
def main(
    model_repo_dir: str,
    output_dir: Path,
    data_dir: Path,
    log_dir: Path,
    image: str,
    model_name: str,
    accounting_group: str,
    accounting_group_user: str,
    Tb: float,
    sample_rate: float,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    fduration: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    verbose: bool = False,
):
    # _fix_path()  # TODO: replace this
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "infer.deploy.log", verbose)

    # get ip address and add to arguments
    # along with the timeslide datadir which will be read from a text file
    ip = get_ip_address()
    log_pattern = "infer-$(ProcID).log"
    output_pattern = "tmp/output-$(ProcID)"
    arguments = f"""
    --data-dir {data_dir}
    --output-dir {output_dir / output_pattern}
    --start $(start)
    --sequence-id $(seq_id)
    --log-file {log_dir / log_pattern}
    --ip {ip}
    --model-name {model_name}
    --sample-rate {sample_rate}
    --inference-sampling-rate {inference_sampling_rate}
    --ifos {" ".join(ifos)}
    --batch-size {batch_size}
    --integration-window-length {integration_window_length}
    --cluster-window-length {cluster_window_length}
    --fduration {fduration}
    --throughput {throughput}
    --chunk-size {chunk_size}
    --model-version {model_version}
    """
    arguments = dedent(arguments).replace("\n", " ")
    if verbose:
        arguments += " --verbose"

    condor_dir = output_dir / "condor"
    condor_dir.mkdir(exist_ok=True, parents=True)

    with h5py.File(data_dir / "background.hdf", "r") as f:
        starts = list(f[ifos[0]].keys())

    parameters = "start,seq_id\n"
    for i, start in enumerate(starts):
        seq_id = sequence_id + 2 * i
        parameters += f"{start},{seq_id}\n"

    submit_file = condor.make_submit_file(
        executable="mdc",
        name="mdc-infer",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condor_dir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory="6G",
        request_disk="1G",
    )

    # spin up triton server
    logging.info("Launching triton server")
    server_log = log_dir / "server.log"
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
            dag_id = condor.submit(submit_file)
            condor.watch(dag_id, condor_dir)

    logging.info("Aggregating results")
    aggregate_results(output_dir)


if __name__ == "__main__":
    main()
