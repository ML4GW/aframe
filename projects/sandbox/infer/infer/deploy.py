import logging
import re
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import List

import psutil
from typeo import scriptify

from aframe.analysis.ledger.events import EventSet, RecoveredInjectionSet
from aframe.deploy import condor
from aframe.logging import configure_logging
from aframe.utils.timeslides import calc_shifts_required
from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor

re_fname = re.compile(r"([0-9]{10})-([1-9][0-9]*)\.")
logging.getLogger("urllib3").setLevel(logging.WARNING)


def aggregate_results(output_directory: Path):
    """
    Combine results from across segments into a single
    background file and foreground file. Remove the directory
    containing the individual segment results.
    """
    background, foreground = EventSet(), RecoveredInjectionSet()
    for data_dir in (output_directory / "tmp").iterdir():
        bckground = EventSet.read(data_dir / "background.h5")
        frground = RecoveredInjectionSet.read(data_dir / "foreground.h5")

        background.append(bckground)
        foreground.append(frground)

    background.write(output_directory / "background.h5")
    foreground.write(output_directory / "foreground.h5")
    shutil.rmtree(output_directory / "tmp")


def get_num_shifts(data_dir: Path, Tb: float, shift: float) -> int:
    """
    Calculates the number of required time shifts based a set of file
    names formatted in the way of `background.py` in the `datagen`
    project.
    """
    T = 0
    for fname in data_dir.iterdir():
        match = re_fname.search(fname.name)
        if match is not None:
            duration = match.group(2)
            T += float(duration)
    return calc_shifts_required(Tb, T, shift)


def get_ip_address() -> str:
    """
    Get the local, cluster-internal IP address.
    Currently not a general function.
    """
    nets = psutil.net_if_addrs()
    return nets["enp1s0f0"][0].address


@scriptify
def main(
    model_repo_dir: str,
    output_dir: Path,
    data_dir: Path,
    log_dir: Path,
    injection_set_file: Path,
    image: str,
    model_name: str,
    accounting_group: str,
    accounting_group_user: str,
    Tb: float,
    shifts: List[float],
    sample_rate: float,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    psd_length: float,
    fduration: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    verbose: bool = False,
):
    """
    Deploy condor jobs to perform inference using Triton
    on a directory of timeseries files, using a particular
    set of interferometer time shifts. Network outputs will
    be saved both as-is and using local integration.

    Args:
        model_repo_dir:
            Directory containing the model repository.
        output_dir:
            Directory to which to save raw and locally
            integrated network outputs.
        data_dir:
            Directory containing input files representing
            timeseries on which to perform inference.
            Each HDF5 file in this directory will be used
            for inference.
        log_dir:
            Directory to which log files will be written.
        injection_set_file:
            Path to a file which can be read as a
            `LigoResponseSet`. Contains the waveforms that
            will be injected into the background.
        image:
            Path to an image of tritonserver
        accounting_group:
            Accounting group for the condor jobs
        accounting_group_user:
            Username of the person running the condor jobs
        Tb:
            The length of background time in seconds to be generated via
            time shifts
        shifts:
            A list of shifts in seconds. Each value corresponds to the
            the length of time by which an interferometer's timeseries is
            moved during one shift. For example, if `ifos = ["H1", "L1"]`
            and `shifts = [0, 1]`, then the Livingston timeseries will be
            advanced by one second per shift, and Hanford won't be shifted
        sample_rate:
            Rate at which input timeseries data has been sampled,
            specified in Hz
        inference_sampling_rate:
            The rate at which to sample windows for inference from
            the input timeseries, specified in Hz.
            Corresponds to the sample rate of the output timeseries.
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        integration_window_length:
            Length of the window over which network
            outputs should be locally integrated,
            specified in seconds.
        cluster_window_length:
            Length of the window over which network
            outputs should be clustered, specified
            in seconds
        psd_length:
            Length of background to use for PSD calculation,
            specified in seconds.
        fduration:
            Length of the time-domain whitening filter,
            specified in seconds.
        throughput:
            Rate at which to make requests, in units
            of seconds of data per second `[s' / s]`.
        chunk_size:
            Length of data to load at once, specified in seconds
        sequence_id:
            Identifier to assign to all the sequences
            of inference requests in this run to match up
            with a corresponding snapshot state on the
            inference server.
        model_version:
            Version of the model from which to request
            inference. Default value of `-1` indicates
            the latest available version of the model.
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
    """
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
    --shifts $(shift0) $(shift1)
    --sequence-id $(seq_id)
    --log-file {log_dir / log_pattern}
    --ip {ip}
    --model-name {model_name}
    --injection-set-file {injection_set_file}
    --sample-rate {sample_rate}
    --inference-sampling-rate {inference_sampling_rate}
    --ifos {" ".join(ifos)}
    --batch-size {batch_size}
    --integration-window-length {integration_window_length}
    --cluster-window-length {cluster_window_length}
    --fduration {fduration}
    --throughput {throughput}
    --chunk-size {chunk_size}
    --psd-length {psd_length}
    --model-version {model_version}
    """
    arguments = dedent(arguments).replace("\n", " ")
    if verbose:
        arguments += " --verbose"

    condor_dir = output_dir / "condor"
    condor_dir.mkdir(exist_ok=True, parents=True)

    max_shift = max(shifts)
    num_shifts = get_num_shifts(data_dir, Tb, max_shift)
    parameters = "shift0,shift1,seq_id\n"
    # skip the 0lag shift
    for i in range(num_shifts):
        seq_id = sequence_id + 2 * i
        parameters += f"0,{(i + 1) * max_shift},{seq_id}\n"

    submit_file = condor.make_submit_file(
        executable="infer",
        name="infer",
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
