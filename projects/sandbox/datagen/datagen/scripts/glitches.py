"""
Script that generates a dataset of glitches from omicron triggers.
"""

import configparser
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import datagen.utils.glitches as utils
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries
from omicron.cli.process import main as omicron_main
from typeo import scriptify

from aframe.deploy import condor
from aframe.logging import configure_logging

logging.getLogger("urllib3").setLevel(logging.WARNING)


def fetch(channel: str, start: float, stop: float, verbose: bool = True):
    logging.debug(f"Fetching {channel} from {start} to {stop}")
    try:
        data = TimeSeries.get(channel, start, stop, verbose=verbose)
    except ValueError as e:
        # only catch ValueError if it's due to above issue
        # otherwise raise the error as normal and have
        # condor retry mechanism resolve it
        msg = str(e)
        logging.info(msg)
        if msg.startswith("["):
            logging.warning(f"Skipping segment due to ValueError: {e}")
            return
        else:
            raise e
    return data


def query_glitches(
    trigger_path: Path,
    pad: Tuple[float, float],
    snr_thresh: float,
    sample_rate: float,
    channel: str,
):
    """
    Collect strain data for a list of glitches

    Args:
        trigger_path:
            Path to file produced by omicron that contains trigger information
        outfile:
            Path to file to store queried strain data
        pad:
            Tuple where the first element is amount of data, in seconds,
            to query before trigger time, and second element is
            amount of data to query after trigger time
        snr_thresh:
            snr threshold above which strain data should be queried
        sample_rate:
            rate at which to sample strain data
        channel:
            Channel of the form `{ifo}:{channel_name}`

    Returns Path to outfile

    """

    glitches = []
    snrs = []
    gpstimes = []

    with h5py.File(trigger_path) as f:
        # apply snr thresh
        triggers = f["triggers"][:]
        mask = triggers["snr"][:] > snr_thresh
        triggers = triggers[mask]

    if len(triggers) == 0:
        return [], None, None

    # parse trigger file name to extract start and stop times
    t0, length = utils.parse_omicron_fname(trigger_path)

    # set 'start' and 'stop' so we that we both:
    # 1. aren't querying unnecessary data
    # 2. aren't querying data outside of the segment
    start = max(np.min(triggers["time"]) - pad[0] - 1, t0)
    stop = min(np.max(triggers["time"]) + pad[1] + 1, t0 + length)

    logging.info(
        f"Querying {len(triggers)} triggers from {trigger_path}"
        f" over {stop - start} seconds of data "
    )

    data = fetch(channel, start, stop)
    data = data.resample(sample_rate)

    # reset start / stop based on what was actually found
    start, stop = data.times[0].value, data.times[-1].value

    # query data for each trigger
    for trigger in triggers:
        time = trigger["time"]
        beg, end = time - pad[0], time + pad[1]

        if end > stop or beg < start:
            logging.warning(
                f"Trigger at time {time} is too close to end of segment."
                f" Skipping"
            )
            continue

        try:
            glitch_ts = data.crop(beg, end)
        except ValueError:
            logging.warning(f"Data not available for trigger at time: {time}")
            continue
        else:
            glitches.append(list(glitch_ts.value))
            snrs.append(trigger["snr"])
            gpstimes.append(time)

    return glitches, snrs, gpstimes


@scriptify
def collect_glitches(
    trigger_path: Path,
    outfile: Path,
    pad: Tuple[float, float],
    snr_thresh: float,
    sample_rate: float,
    channel: str,
):
    glitches, snrs, times = query_glitches(
        trigger_path, pad, snr_thresh, sample_rate, channel
    )

    if glitches:
        glitches = np.stack(glitches)
        with h5py.File(outfile, "w") as f:
            f.create_dataset("glitches", data=glitches)
            f.create_dataset("snrs", data=snrs)
            f.create_dataset("times", data=times)

    return outfile


def deploy_collect_glitches(
    trigger_paths: List[Path],
    pad: Tuple[float, float],
    snr_thresh: float,
    sample_rate: float,
    channel: str,
    ifo: str,
    outdir: Path,
    condordir: Path,
    accounting_group: str,
    accounting_group_user: str,
    request_memory: float = 32768,
):
    """
    Deploys a fleet of condor jobs to collect strain data
    from multiple trigger files in parallel
    Args:
        trigger_paths:
            List of trigger files. A condor job will be launched for each
            trigger file
        pad:
            Tuple where the first element is amount of data, in seconds,
            to query before trigger time, and second element
            is amount of data to query after trigger time
        snr_thresh:
            snr threshold above which strain data should be queried
        sample_rate:
            rate at which to sample strain data
        channel:
            Channel of the form `{channel_name}`
        ifo:
            IFO to query data for
        outdir:
            Directory where glitch strain data files will be stored
        condordir:
            Location where condor related files will be stored
        accounting_group:
            Accounting group for the condor jobs
        accounting_group_user:
            Username of the person running the condor jobs
        request_memory:
            Amount of initial memory to request for each condor job
    """
    condordir = condordir / ifo
    condordir.mkdir(exist_ok=True, parents=True)
    outdir.mkdir(exist_ok=True, parents=True)
    # create text file from which the condor job will read
    # the trigger_path, and outfile for each job
    parameters = "trigger_path,outfile\n"

    for f in trigger_paths:
        t0, length = utils.parse_omicron_fname(f)
        logging.debug(f"Generating glitch dataset for {ifo} and file {f}")
        out = outdir / f"{ifo}-glitches-{t0}-{length}.hdf5"
        if out.exists():
            logging.info(f"Found existing glitch file {out}. Skipping")
            continue
        parameters += f"{f},{out}\n"

    arguments = "--trigger-path $(trigger_path) --outfile $(outfile) "
    arguments += f"--pad {pad[0]} {pad[1]} --snr-thresh {snr_thresh} "
    arguments += f"--channel {channel} --sample-rate {sample_rate} "

    subfile = condor.make_submit_file(
        executable="collect-glitches",
        name="collect_glitches",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory=request_memory,  # noqa
        periodic_release="(HoldReasonCode =?= 26 || HoldReasonCode =?= 34) && (JobStatus == 5)",  # noqa
        periodic_remove="(JobStatus == 1) && MemoryUsage >= 7G",
        max_retries=5,
    )

    dag_id = condor.submit(subfile)
    logging.info(
        f"Launching collection of glitches for {ifo} with dag id {dag_id}"
    )
    condor.watch(dag_id, condordir, held=False)
    logging.info(f"Completed collection of glitches for {ifo} ")

    return outdir


def omicron_main_wrapper(
    start: int,
    stop: int,
    run_dir: Path,
    q_min: float,
    q_max: float,
    f_min: float,
    f_max: float,
    sample_rate: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    snr_thresh: float,
    frame_type: str,
    channel: str,
    state_flag: str,
    ifo: str,
    log_file: Path,
    verbose: bool,
):

    """
    Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file

    config = configparser.ConfigParser()
    section = "GW"
    config.add_section(section)

    config.set(section, "q-range", f"{q_min} {q_max}")
    config.set(section, "frequency-range", f"{f_min} {f_max}")
    config.set(section, "frametype", f"{ifo}_{frame_type}")
    config.set(section, "channels", f"{ifo}:{channel}")
    config.set(section, "cluster-dt", str(cluster_dt))
    config.set(section, "sample-frequency", str(sample_rate))
    config.set(section, "chunk-duration", str(chunk_duration))
    config.set(section, "segment-duration", str(segment_duration))
    config.set(section, "overlap-duration", str(overlap))
    config.set(section, "mismatch-max", str(mismatch_max))
    config.set(section, "snr-threshold", str(snr_thresh))
    # in an online setting, can also pass state-vector,
    # and bits to check for science mode
    config.set(section, "state-flag", f"{ifo}:{state_flag}")

    config_file_path = run_dir / f"omicron_{ifo}.ini"

    # write config file
    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    # parse args into format expected by omicron
    omicron_args = [
        section,
        "--log-file",
        str(log_file),
        "--config-file",
        str(config_file_path),
        "--gps",
        f"{start}",
        f"{stop}",
        "--ifo",
        ifo,
        "-c",
        "request_disk=4GB",
        "--output-dir",
        str(run_dir),
        "--skip-gzip",
        "--skip-rm",
        # see https://git.ligo.org/computing/helpdesk/-/issues/4512
        # wait for https://github.com/gwpy/pyomicron/pull/164
        "-d",
        "include_env=X509_USER_PROXY,PATH",
    ]
    if verbose:
        omicron_args += ["--verbose"]

    # create and launch omicron dag
    omicron_main(omicron_args)

    # return variables necessary for glitch generation
    return ifo


@scriptify
def main(
    snr_thresh: float,
    start: int,
    stop: int,
    test_stop: int,
    q_min: float,
    q_max: float,
    f_min: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    datadir: Path,
    logdir: Path,
    channel: str,
    frame_type: str,
    sample_rate: float,
    state_flag: str,
    ifos: List[str],
    fduration: float,
    psd_length: float,
    kernel_length: float,
    accounting_group: str,
    accounting_group_user: str,
    pad: Optional[Tuple[float, float]] = None,
    analyze_testing_set: bool = False,
    force_generation: bool = False,
    verbose: bool = False,
):

    """
    Generate a set of glitches for both
    H1 and L1 that can be added to background.

    First, omicron jobs are launched via pyomicron
    (https://github.com/gwpy/pyomicron/). Next, triggers (i.e. glitches)
    above a given SNR threshold are selected from the omicron output,
    and data is queried for these triggers and saved in hdf5 files.

    These files contains a group for each interferometer in `ifos`.
    Within each group, the `times` dataset contains the GPS time
    of each glitch, the `snrs` dataset contains the SNR of each glitch,
    and the `glitches` dataset contains the actual strain data


    Args:
        snr_thresh:
            SNR threshold above which glitches will be kept
        start:
            GPS time at which to begin looking for glitches
        stop:
            GPS time at which to stop looking for glitches for the
            training dataset.
            Marks the beginning of the testing dataset
        test_stop:
            GPS time at which to stop looking for glitches for the
            testing dataset
        q_min:
            Minimum q value of tiles for omicron
        q_max:
            Maximum q value of tiles for omicron
        f_min:
            Lowest frequency for omicron to consider, specified in Hz
        cluster_dt:
            Time window for omicron to cluster neighboring triggers
        chunk_duration:
            Duration of data in seconds for PSD estimation
        segment_duration:
            Duration of data in seconds for FFT
        overlap:
            Overlap in seconds between neighbouring segments and chunks
        mismatch_max:
            Maximum distance between (Q, f) tiles
        datadir:
            Directory to which the glitch dataset will be written
        logdir:
            Directory to which the log file will be written
        channel:
            Channel name used to read data. Should not include the
            interferometer prefix
        frame_type:
            Frame type for data discovery with gwdatafind
        sample_rate:
            Sample rate of queried data, specified in Hz
        state_flag:
            Identifier for which segments to use. Descriptions of flags
            and there usage can be found here:
            https://wiki.ligo.org/DetChar/DataQuality/AligoFlags
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        fduration:
            Duration of the time domain filter used
            to whiten the data as a preprocessing step.
            Used here to determine quantity of data around glitch
            to query
        psd_length:
            The length, in seconds, of data used to whiten during
            training. Used here to determine quantity of data around glitch
            to query
        kernel_length:
            The length, in seconds, of each window of data
            (after cropping from whitening) that the
            neural network will analyze. Used here to determine
            quantity of data around glitch to query
        accounting_group:
            Accounting group for the condor jobs
        accounting_group_user:
            Username of the person running the condor jobs
        analyze_testing_set:
            If True, get glitches for the testing dataset
        force_generation:
            If False, will not generate data if an existing dataset exists
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.

    Returns: The name of the file containing the glitch data
    """

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    log_file = logdir / "glitches.log"
    configure_logging(log_file, verbose)

    # AFAIK pyomicron has no mechanism for querying
    # open data. So, if user specifies use of open data,
    # manually choose correct frame type, state flag, and channel
    channel = utils.get_channel(channel)
    state_flag = utils.get_state_flag(state_flag)

    # TODO: implement some sort of caching mechanism

    # nyquist
    f_max = sample_rate / 2

    run_dir = datadir / "condor" / "omicron"
    train_run_dir = run_dir / "train"
    test_run_dir = run_dir / "test"
    omicron_log_file = run_dir / "pyomicron.log"

    # seconds before and after trigger time such that we
    # we have exactly enough data to calculate psd for whitening, and sample
    # the trigger time at the first or last sample of the kernel
    if pad is None:
        pad = (
            psd_length + kernel_length + (fduration / 2),
            kernel_length + (fduration / 2),
        )

    # spin up pool to parallelize launching
    # of (condor based) pyomicron jobs for
    # both testing and training segments
    pool = ThreadPoolExecutor(4)
    omicron_futures = []

    for ifo in ifos:
        train_ifo_dir = train_run_dir / ifo
        test_ifo_dir = test_run_dir / ifo
        train_ifo_dir.mkdir(exist_ok=True, parents=True)

        # launch pyomicron futures for training set and testing sets.
        # as train futures complete, launch glitch generation processes.
        # let test set jobs run in background as glitch data is queried
        args = [
            q_min,
            q_max,
            f_min,
            f_max,
            sample_rate,
            cluster_dt,
            chunk_duration,
            segment_duration,
            overlap,
            mismatch_max,
            snr_thresh,
            frame_type,
            channel,
            state_flag,
            ifo,
            omicron_log_file,
            verbose,
        ]

        future = pool.submit(
            omicron_main_wrapper, start, stop, train_ifo_dir, *args
        )
        omicron_futures.append(future)

        # might be interested in glitch times for testing set
        if analyze_testing_set:
            test_ifo_dir.mkdir(exist_ok=True, parents=True)
            pool.submit(
                omicron_main_wrapper, stop, test_stop, test_ifo_dir, *args
            )

    # as omicron futures complete, deploy glitch collection
    # condor jobs that will query the actual strain data.
    collect_futures = []
    for future in as_completed(omicron_futures):
        ifo = utils.handle_future(future)
        outdir = datadir / "train" / "glitches" / ifo
        condordir = datadir / "condor" / "glitches"
        trigger_dir = train_run_dir / ifo / "merge" / f"{ifo}:{channel}"
        trigger_paths = sorted(list(trigger_dir.glob("*.h5")))

        args = [
            trigger_paths,
            pad,
            snr_thresh,
            sample_rate,
            f"{ifo}:{channel}",
            ifo,
            outdir,
            condordir,
            accounting_group,
            accounting_group_user,
        ]
        future = pool.submit(deploy_collect_glitches, *args)
        collect_futures.append(future)

    for future in as_completed(collect_futures):
        result = utils.handle_future(future)
        logging.info(f"Completed collection of glitches for {result} ")

    return datadir


if __name__ == "__main__":
    main()
