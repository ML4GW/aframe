import logging
from pathlib import Path
from typing import List, Tuple

import h5py
from mldatafind.authenticate import authenticate
from mldatafind.io import fetch_timeseries
from mldatafind.segments import query_segments
from typeo import scriptify

from aframe.deploy import condor
from aframe.logging import configure_logging


def _intify(x: float):
    return int(x) if int(x) == x else x


def _make_fname(prefix, t0, length):
    t0 = _intify(t0)
    length = _intify(length)
    return f"{prefix}-{t0}-{length}.hdf5"


def validate_file(
    filename: Path,
    ifos: List[str],
    sample_rate: float,
    start: float,
    stop: float,
    minimum_length: float,
):
    # if there exists files in training range,
    # check the timestamp and verify that it
    # meets the requested conditions
    with h5py.File(filename, "r") as f:
        missing_keys = [i for i in ifos if i not in f]
        if missing_keys:
            raise ValueError(
                "Background file {} missing data from {}".format(
                    filename, ", ".join(missing_keys)
                )
            )

        x = f[ifos[0]]
        t0 = x.attrs["x0"][()]
        length = len(x) / sample_rate

    in_range = start <= t0 <= (stop - minimum_length)
    long_enough = length >= minimum_length
    if not (in_range and long_enough):
        raise ValueError(
            "Background file {} has t0 {} and length {}s, "
            "which isn't compatible with request of {}s "
            "segment between {} and {}".format(
                filename,
                t0,
                length,
                minimum_length,
                start,
                stop,
            )
        )


def split_segments(segments: List[tuple], chunk_size: float):
    out_segments = []
    for segment in segments:
        start, stop = segment
        duration = stop - start
        if duration > chunk_size:
            num_segments = int((duration - 1) // chunk_size) + 1
            logging.info(f"Chunking segment into {num_segments} parts")
            for i in range(num_segments):
                end = min(start + (i + 1) * chunk_size, stop)
                seg = (start + i * chunk_size, end)
                out_segments.append(seg)
        else:
            out_segments.append(segment)
    return out_segments


def validate_segments(
    segments: List[Tuple[float, float]],
    train_start: float,
    train_stop: float,
    test_stop: float,
    minimum_train_length: float,
    minimum_test_length: float,
    max_segment_length: float,
    datadir: Path,
    force_generation: bool,
    ifos: List[str],
    sample_rate: float,
):
    segments = split_segments(segments, max_segment_length)
    validated = []
    for start, stop in segments:

        # using start/stops to decide if something
        # is a training file or not to make robust
        # to future use of multiple training background
        is_train = train_start <= start <= (train_stop - minimum_train_length)
        if is_train:
            subdir = "train"
            stop = min(stop, train_stop)
        else:
            subdir = "test"

        write_dir = datadir / subdir / "background"
        write_dir.mkdir(parents=True, exist_ok=True)
        fname = _make_fname("background", start, stop - start)
        write_path = write_dir / fname

        if write_path.exists() and not force_generation:
            if is_train:
                validate_file(
                    write_path,
                    ifos,
                    sample_rate,
                    train_start,
                    train_stop,
                    minimum_train_length,
                )
            else:
                validate_file(
                    write_path,
                    ifos,
                    sample_rate,
                    train_stop,
                    test_stop,
                    minimum_test_length,
                )

            logging.info(
                "Skipping download of segment {}-{}, already "
                "cached in file {}".format(start, stop, fname)
            )
            continue

        logging.info(
            "Adding condor job to download segment {}-{} to file {}".format(
                start, stop, write_path
            )
        )
        validated.append([start, stop, write_path])
    return validated


@scriptify
def main(
    start: float,
    stop: float,
    writepath: Path,
    channel: str,
    ifos: List[str],
    sample_rate: float,
):
    """Generates background data for training and testing aframe

    Args:
        start: start gpstime
        stop: stop gpstime
        ifos: which ifos to query data for
        outdir: where to store data
    """
    authenticate()
    channels = [f"{ifo}:{channel}" for ifo in ifos]
    data = fetch_timeseries(channels, start, stop)
    data = data.resample(sample_rate)
    for ifo in ifos:
        data[ifo] = data.pop(f"{ifo}:{channel}")

    data.write(writepath)
    return writepath


@scriptify
def deploy(
    train_start: float,
    train_stop: float,
    test_stop: float,
    minimum_train_length: float,
    minimum_test_length: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    state_flag: str,
    datadir: Path,
    logdir: Path,
    accounting_group: str,
    accounting_group_user: str,
    # note that doing consecutive runs while changing max_segment_length
    # will screw with the caching checking, so be careful
    max_segment_length: float = 20000,
    request_memory: int = 8192,
    request_disk: int = 1024,
    force_generation: bool = False,
    verbose: bool = False,
):

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)
    condordir = datadir / "condor" / "background"
    condordir.mkdir(exist_ok=True, parents=True)
    configure_logging(str(logdir / "generate_background.log"), verbose)

    # first query segments that meet minimum length
    # requirement during the requested training period
    # authenticate()
    train_segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        train_start,
        train_stop,
        minimum_train_length,
    )
    if not train_segments:
        raise ValueError(
            "No segments of minimum length, not producing background"
        )

    test_segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        train_stop,
        test_stop,
        minimum_test_length,
    )

    segments = train_segments + test_segments

    # determine which segments we need to generate data for
    segments = validate_segments(
        segments,
        train_start,
        train_stop,
        test_stop,
        minimum_train_length,
        minimum_test_length,
        max_segment_length,
        datadir,
        force_generation,
        ifos,
        sample_rate,
    )

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,writepath\n"
    for start, stop, writepath in segments:
        parameters += f"{start},{stop},{writepath}\n"

    arguments = "--start $(start) --stop $(stop) "
    arguments += "--writepath $(writepath) "
    arguments += f"--channel {channel} --sample-rate {sample_rate} "
    arguments += f"--ifos {' '.join(ifos)} "

    kwargs = {"+InitialRequestMemory": request_memory}
    subfile = condor.make_submit_file(
        executable="generate-background",
        name="generate_background",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_disk=request_disk,
        # stolen from pyomicron: allows dynamic updating of memory
        request_memory=f"ifthenelse(isUndefined(MemoryUsage), {request_memory}, int(3*MemoryUsage))",  # noqa
        periodic_release="(HoldReasonCode =?= 26 || HoldReasonCode =?= 34) && (JobStatus == 5)",  # noqa
        periodic_remove="(JobStatus == 1) && MemoryUsage >= 7G",
        use_x509userproxy=True,
        **kwargs,
    )
    dag_id = condor.submit(subfile)
    logging.info(f"Launching background generation jobs with dag id {dag_id}")
    condor.watch(dag_id, condordir, held=False)
    logging.info("Completed background generation jobs")
