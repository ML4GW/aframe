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
    """
    Converts the input float into an int if the two are equal (e.g., 4.0 == 4).
    Otherwise, returns the input unchanged.
    """
    return int(x) if int(x) == x else x


def _make_fname(prefix, t0, length):
    """Creates a filename for background files in a consistent format"""
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
    """
    If there exist files in the time range, check the timestamp
    and verify that it meets the requested conditions
    """
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
    """
    Split a list of segments into segments that are at most
    `chunk_size` seconds long
    """
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
    """
    Check whether any background data for a set of segments already exists.

    This check depends on how the segments are split using
    `max_segment_length`. It may be that equivalent data already exists
    in the data directory, but spread differently over the data files.
    For the check to be accurate, the same `max_segment_length` needs
    to be used.

    Returns a list of segments that there is not data for.
    """
    segments = split_segments(segments, max_segment_length)
    validated = []
    for start, stop in segments:
        duration = stop - start
        # using start/stops to decide if something
        # is a training file or not to make robust
        # to future use of multiple training background
        is_train = train_start <= start
        is_train &= stop <= train_stop

        if is_train:
            subdir = "train"
            stop = min(stop, train_stop)
            if duration < minimum_train_length:
                logging.info(
                    "Skipping segment {}-{}, too short for training".format(
                        start, stop
                    )
                )
                continue
        else:
            subdir = "test"
            stop = min(stop, test_stop)
            if duration < minimum_test_length:
                logging.info(
                    "Skipping segment {}-{}, too short for testing".format(
                        start, stop
                    )
                )
                continue

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
        start:
            Starting GPS time of the timeseries to be fetched
        stop:
            Ending GPS time of the timeseries to be fetched
        writepath:
            Path, including file name, that the data will be saved to
        channel:
            Channel from which to fetch the timeseries
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        sample_rate:
            Sample rate to which the timeseries will be resampled

    Returns: The `Path` of the output file
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
    max_segment_length: float = 20000,
    request_memory: int = 32768,
    request_disk: int = 1024,
    force_generation: bool = False,
    verbose: bool = False,
):
    """
    Deploy condor jobs to download background data.

    Args:
        train_start:
            GPS time of the beginning of the training period.
        train_stop:
            GPS time of the end of the training period.
            Also corresponds to the beginning of the testing period.
        test_stop:
            GPS time of the end of the testing period.
        minimum_train_length:
            The shortest a contiguous segment of training background can be.
            Specified in seconds.
        minimum_test_length:
            The shortest a contiguous segment of testing background can be.
            Specified in seconds.
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        sample_rate:
            Sample rate to which the timeseries will be resampled
        channel:
            Channel from which to fetch the timeseries
        state_flag:
            Identifier for which segments to use
        datadir:
            Directory to which data will be written
        logdir:
            Directory to which the log file will be written
        accounting_group_user:
            Username of the person running the condor jobs
        accounting_group:
            Accounting group for the condor jobs
        max_segment_length:
            Maximum length of a segment in seconds. Note that doing
            consecutive runs while changing `max_segment_length` will
            screw with the caching checking, so be careful.
        request_memory:
            Amount of memory for condor jobs to request in Mb
        request_disk:
            Amount of disk space for condor jobs to request in Mb
        force_generation:
            If false, will not generate data if an existing dataset exists
        verbose:
            If true, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
    """
    # make logdir dir
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

    segments = list(train_segments) + list(test_segments)

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

    if not segments:
        logging.info("No segments to generate, not deploying condor jobs")
        return

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,writepath\n"
    for start, stop, writepath in segments:
        parameters += f"{start},{stop},{writepath}\n"

    arguments = "--start $(start) --stop $(stop) "
    arguments += "--writepath $(writepath) "
    arguments += f"--channel {channel} --sample-rate {sample_rate} "
    arguments += f"--ifos {' '.join(ifos)} "

    if verbose:
        arguments += "--verbose "

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
    condor.watch(dag_id, condordir, held=True)
    logging.info("Completed background generation jobs")
