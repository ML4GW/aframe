import logging
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, wait
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import h5py
import numpy as np
from rich.progress import Progress
from utils import characterize_events, get_write_dir, load_segments

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.distributions.cluster import ClusterDistribution
from bbhnet.analysis.normalizers import GaussianNormalizer
from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed
from hermes.typeo import typeo

if TYPE_CHECKING:
    from bbhnet.analysis.distributions.distribution import Distribution


def build_background(
    thread_ex: AsyncExecutor,
    process_ex: AsyncExecutor,
    pbar: Progress,
    background_segments: Iterable[Segment],
    data_dir: Path,
    write_dir: Path,
    max_tb: float,
    t_clust: float,
    window_length: float = 1.0,
    norm_seconds: Optional[Iterable[float]] = None,
):
    """
    For a sequence of background segments, compute a discrete
    distribution of integrated neural network outputs using
    the indicated integration window length for each of the
    normalization window lengths specified. Iterates through
    the background segments in order and tries to find as
    many time-shifts available for each segment as possible
    in the specified data directory, stopping iteration through
    segments once a maximum number of seconds of bacgkround have
    been generated.
    As a warning, there's a fair amount of asynchronous execution
    going on in this function, and it may come off a bit complex.
    Args:
        thread_ex:
            An `AsyncExecutor` that maintains a thread pool
            for writing analyzed segments in parallel with
            the analysis processes themselves.
        process_ex:
            An `AsyncExecutor` that maintains a process pool
            for loading and integrating Segments of neural
            network outputs.
        pbar:
            A `rich.progress.Progress` object for keeping
            track of the progress of each of the various
            subtasks.
        background_segments:
            The `Segment` objects to use for building a
            background distribution. `data_dir` will be
            searched for all time-shifts of each segment
            for parallel analysis. Once `max_tb` seconds
            worth of background have been generated, iteration
            through this array will be terminated, so segments
            should be ordered by some level of "importance",
            since it's likely that segments near the back of the
            array won't be analyzed for lower values of `max_tb`.
        data_dir:
            Directory containing timeslide root directories,
            which will be mined for time-shifts of each `Segment`
            in `background_segments`. If a time-shift doesn't exist
            for a given `Segment`, the time-shift is ignored.
        write_dir:
            Root directory to which to write integrated NN outputs.
            For each time-shift analyzed and normalization window
            length specified in `norm_seconds`, results will be
            written to a subdirectory
            `write_dir / "norm-seconds.{norm}" / shift`, which
            will be created if it does not exist.
        max_tb:
            The maximum number of seconds of background data
            to analyze for each value of `norm_seconds` before
            new segments to shift and analyze are no longer sought.
            However, because we use _every_ time-shift for each
            segment we iterate through, its possible that each
            background distribution will utilize slightly more
            than this value.
        window_length:
            The length of the integration window to use
            for analysis in seconds.
        t_clust:
            Timescale over which to cluster integrated network outputs
    Returns:
        A dictionary mapping each value in `norm_seconds` to
            an associated `ClusterDistribution` characterizing
            its background distribution.
    """

    write_dir.mkdir(exist_ok=True)
    norm_seconds = norm_seconds or [norm_seconds]

    # keep track of all the files that we've written
    # for each normalization window size so that we
    # can iterate through them later and submit them
    # for reloading once we have our distributions initialized
    fname_futures = defaultdict(list)

    # iterate through timeshifts of our background segments
    # until we've generated enough background data.
    background_segments = iter(background_segments)
    main_task_id = pbar.add_task("[red]Building background", total=max_tb)
    while not pbar.tasks[main_task_id].finished:
        segment = next(background_segments)

        # since we're assuming here that the background
        # segments are being provided in reverse chronological
        # order (with segments closest to the event segment first),
        # exhaust all the time shifts we can of each segment before
        # going to the previous one to keep data as fresh as possible
        load_futures = {}
        for shift in data_dir.iterdir():
            try:
                shifted = segment.make_shift(shift.name)
            except ValueError:
                # this segment doesn't have a shift
                # at this value, so just move on
                continue

            # load all the timeslides up front in a separate thread
            # TODO: O(1GB) memory means segment.length * N ~O(4M),
            # so for ~O(10k) long segments this means this should
            # be fine as long as N ~ O(100). Worth doing a check for?
            future = process_ex.submit(load_segments, shifted, "out")
            load_futures[shift.name] = [future]

        # create progress bar tasks for each one
        # of the subprocesses involved for analyzing
        # this set of timeslide ClusterDistributions
        load_task_id = pbar.add_task(
            f"[cyan]Loading {len(load_futures)} {segment.length}s timeslides",
            total=len(load_futures),
        )
        analyze_task_id = pbar.add_task(
            "[yelllow]Integrating timeslides",
            total=len(load_futures) * len(norm_seconds),
        )
        write_task_id = pbar.add_task(
            "[green]Writing integrated timeslides",
            total=len(load_futures) * len(norm_seconds),
        )

        # now once each segment is loaded, submit a job
        # to our process pool to integrate it using each
        # one of the specified normalization periods
        integration_futures = {}
        sample_rate = None
        for shift, seg in as_completed(load_futures):
            # get the sample rate of the NN output timeseries
            # dynamically from the first timeseries we load,
            # since we'll need it to initialize our normalizers
            if sample_rate is None:
                t = seg._cache["t"]
                sample_rate = 1 / (t[1] - t[0])

            # 'load' the network output
            # and times for this segment;
            # as they are already cached,
            # this should just return them
            y, t = seg.load("out")

            for norm in norm_seconds:
                # build a normalizer for the given normalization window length
                if norm is not None:
                    normalizer = GaussianNormalizer(norm * sample_rate)
                    # fit the normalizer to the
                    # background segment
                    normalizer.fit(y)

                else:
                    normalizer = None

                # submit the integration job and have it update the
                # corresponding progress bar task once it completes
                future = process_ex.submit(
                    integrate,
                    y,
                    t,
                    kernel_length=1.0,
                    window_length=window_length,
                    normalizer=normalizer,
                )
                future.add_done_callback(
                    lambda f: pbar.update(analyze_task_id, advance=1)
                )
                integration_futures[(norm, shift)] = [future]

            # advance the task keeping track of how many files
            # we've loaded by one
            pbar.update(load_task_id, advance=1)

        # make sure we have the expected number of jobs submitted
        if len(integration_futures) < (len(norm_seconds) * len(load_futures)):
            raise ValueError(
                "Expected {} integration jobs submitted, "
                "but only found {}".format(
                    len(norm_seconds) * len(load_futures),
                    len(integration_futures),
                )
            )

        # as the integration jobs come back, write their
        # results using our thread pool and record the
        # min and max values for our discrete distribution
        segment_futures = []
        for (norm, shift), (t, y, integrated) in as_completed(
            integration_futures
        ):
            # submit the writing job to our thread pool and
            # use a callback to keep track of all the filenames
            # for a given normalization window
            shift_dir = get_write_dir(
                write_dir, shift, "background-integrated", norm
            )
            future = thread_ex.submit(
                write_timeseries,
                shift_dir,
                t=t,
                y=y,
                integrated=integrated,
            )
            future.add_done_callback(
                lambda f: pbar.update(write_task_id, advance=1)
            )
            fname_futures[norm].append(future)
            segment_futures.append(future)

        # wait for all the writing to finish before we
        # move on so that we don't overload our processes
        wait(segment_futures, return_when=FIRST_EXCEPTION)
        pbar.update(main_task_id, advance=len(load_futures) * segment.length)

    Tb = pbar.tasks[main_task_id].completed
    logging.info(f"Accumulated {Tb}s of background matched filter outputs.")

    # submit a bunch of jobs for loading these integrated
    # segments back in.
    # TODO: don't need to reload in segments
    load_futures = defaultdict(list)
    for norm, fname in as_completed(fname_futures):
        future = process_ex.submit(load_segments, Segment(fname), "integrated")
        load_futures[norm].append(future)

    # create a task for each one of the normalization windows
    # tracking how far along the distribution fit is
    fit_task_ids = {}
    for norm in norm_seconds:
        norm_name = f"{norm}s" if norm is not None else "empty"
        task_id = pbar.add_task(
            "[purple]Fitting background using {} normalization window".format(
                norm_name
            ),
            total=len(load_futures[norm]),
        )
        fit_task_ids[norm] = task_id

    # now discretized the analyzed segments as they're loaded back in
    backgrounds = {}
    for norm, segment in as_completed(load_futures):
        try:
            # if we already have a background distribution
            # for this event, grab it and fit it with a
            # "warm start" aka don't ditch the existing histogram
            background = backgrounds[norm]
            warm_start = True
        except KeyError:
            # otherwise create a new distribution
            # and fit it from scratch
            background = ClusterDistribution("integrated", t_clust)
            backgrounds[norm] = background
            warm_start = False

        # fit the distribution to the new data and then
        # update the corresponding task tracker
        background.fit(segment, warm_start=warm_start)
        pbar.update(fit_task_ids[norm], advance=1)

    for norm, background in backgrounds.items():
        background.write(write_dir / f"background_{norm}.h5")

    return backgrounds, sample_rate


def analyze_injections(
    thread_ex: AsyncExecutor,
    process_ex: AsyncExecutor,
    pbar: Progress,
    injection_segments: Iterable[Segment],
    background_segments: Iterable[Segment],
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    backgrounds: Dict[str, "Distribution"],
    sample_rate: float,
    window_length: float = 1.0,
    kernel_length: float = 1.0,
    metric: str = "far",
):
    """Analyzes a set of events injected on top of timeslides

    thread_ex:
        An `AsyncExecutor` that maintains a thread pool
        for writing analyzed segments in parallel with
        the analysis processes themselves.
    process_ex:
        An `AsyncExecutor` that maintains a process pool
        for loading and integrating Segments of neural
        network outputs.
    pbar:
        A `rich.progress.Progress` object for keeping
        track of the progress of each of the various
        subtasks.

    injection_segments:
        List of segments with injections to analyze
    background_segments:
        List of segments to use for normalization. Expected that
        the background segment corresponds to the data used to
        create the injection segment
    data_dir:
        Directory containing timeslide root directories,
        which will be mined for time-shifts of each `Segment`
        in `injection_segments`. If a time-shift doesn't exist
        for a given `Segment`, the time-shift is ignored.
    write_dir:
        Root directory to write integrated injection segments to
    results_dir:
        Directory to store results from analysis of event times
    backgrounds:
        Dictionary of background Distribution objects where key is
        the normalization in seconds and value is a Distribution object
    sample_rate:
        sample rate of data
    window_length:
        The length of time, in seconds, over which previous
        network outputs should be averaged to produce
        "matched filter" outputs. If left as `None`, it will
        default to the same length as the kernel length.
    kernel_length:
        Length of time of kernel passed to the network
    metric:
        metric to use for characterizing events; "far" or "significance"
    """

    # analyze injections
    # for each normalization length
    # in backgrounds dictionary

    for norm, background in backgrounds.items():
        master_fars, master_latencies, master_event_times = [], [], []

        # create normalizer
        if norm is not None:
            normalizer = GaussianNormalizer(norm * sample_rate)
        else:
            normalizer = None

        # get all the timeslide directories
        shifts = list(data_dir.iterdir())

        main_task_id = pbar.add_task(
            f"[red]Analyzing injections with norm {norm}",
            total=len(background_segments),
        )

        logging.info(
            f"Analyzing {len(injection_segments)}"
            f" with {len(shifts)} timeslides"
        )

        # loop over injection segments
        # and the corresponding background segment
        # the injections were added into;
        # use this background segment to normalize
        # the injection segments

        for back_seg, injection_seg in zip(
            background_segments, injection_segments
        ):

            # for this segment, establish
            # window where events can lie;
            # can't analyze events that don't have norm seconds
            # of data before them
            if norm is not None:
                event_window = (injection_seg.t0 + norm, injection_seg.tf)
            else:
                event_window = (injection_seg.t0, injection_seg.tf)

            # load this injection segment
            # and the corresponding background segment
            # for all timeshits up front;
            load_futures = {}
            for shift in shifts:
                try:
                    back_shifted = back_seg.make_shift(shift.name)
                    injection_shifted = injection_seg.make_shift(shift.name)
                except ValueError:
                    # this segment doesn't have a shift
                    # at this value, so just move on
                    continue

                future = process_ex.submit(
                    load_segments, [back_shifted, injection_shifted], "out"
                )
                load_futures[shift.name] = [future]

            # task id's to track progress for this segment
            load_task_id = pbar.add_task(
                f"[cyan]Loading {2 * len(load_futures)}"
                f" {back_shifted.length}s timeslides",
                total=len(load_futures),
            )
            analyze_task_id = pbar.add_task(
                "[yelllow]Integrating timeslides",
                total=len(load_futures),
            )
            write_task_id = pbar.add_task(
                "[green]Writing integrated timeslides",
                total=len(load_futures),
            )

            characterize_task_id = pbar.add_task(
                "[green]Characterizing events",
                total=len(load_futures),
            )

            # as the segment loading jobs complete
            # submit jobs to integrate the injection segments
            # using the background segment for normalization
            integrate_futures = {}
            for shift, segments in as_completed(load_futures):
                back, injection = segments

                # as segments are already
                # pre-loaded, should just return from cache
                back_y, t = back.load("out")
                injection_y, _ = injection.load("out")

                # fit the normalizer
                # to the background if passed
                if normalizer is not None:
                    normalizer.fit(back_y)

                # submit integration job
                future = process_ex.submit(
                    integrate,
                    injection_y,
                    t,
                    kernel_length=kernel_length,
                    window_length=window_length,
                    normalizer=normalizer,
                )
                future.add_done_callback(
                    lambda f: pbar.update(analyze_task_id, advance=1)
                )

                integrate_futures[shift] = [future]

                pbar.update(load_task_id, advance=2)

            # as the integration jobs come back
            # submit jobs to write integration outputs

            # Also, submit jobs to analyze event times
            # using the integrated outputs
            write_futures = []
            characterize_futures = []
            for shift, (t, y, integrated) in as_completed(integrate_futures):

                # get params file for injections for this timeshift
                param_file = data_dir / shift / "injection" / "params.h5"
                with h5py.File(param_file) as f:

                    # TODO: we seem to be systematically off by 1.5 seconds;
                    # need to find discrepancy
                    event_times = f["geocent_time"][()] + 1.5

                # restrict event times of interest
                # to those within this segment, taking
                # into account normalization
                segment_event_times = [
                    time
                    for time in event_times
                    if time < event_window[1] and time > event_window[0]
                ]

                shift_dir = get_write_dir(
                    write_dir, shift, "injection-integrated", norm
                )

                write_future = thread_ex.submit(
                    write_timeseries,
                    shift_dir,
                    t=t,
                    y=y,
                    integrated=integrated,
                )

                write_futures.append(future)
                write_future.add_done_callback(
                    lambda f: pbar.update(write_task_id, advance=1)
                )

                segment = (integrated, t)

                characterize_future = process_ex.submit(
                    characterize_events,
                    background,
                    segment,
                    segment_event_times,
                    window_length=window_length,
                    metric=metric,
                )

                characterize_future.add_done_callback(
                    lambda f: pbar.update(characterize_task_id, advance=1)
                )
                characterize_futures.append(characterize_future)

            # as event characterizations are done,
            # append results to master lists
            for fars, latencies, times in as_completed(characterize_futures):

                master_fars.append(fars)
                master_latencies.append(latencies)
                master_event_times.append(times)

            master_fars = np.vstack(master_fars)
            master_latencies = np.vstack(master_latencies)
            master_event_times = np.vstack(master_event_times)

            pbar.update(main_task_id, advance=1)

        logging.info(f"Saving analysis data to {results_dir}")
        with h5py.File(results_dir / f"injections-{norm}.h5", "w") as f:
            f.create_dataset("fars", data=master_fars)
            f.create_dataset("latencies", data=master_latencies)
            f.create_dataset("event_times", data=master_event_times)


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    t_clust: float,
    window_length: float = 1.0,
    norm_seconds: Optional[List[float]] = None,
    max_tb: Optional[float] = None,
    force: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    """Analyze injections in a directory of timeslides

    Iterate through a directory of timeslides analyzing known
    injections for false alarm rates in units of yrs$^{-1}$ as
    a function of the time after the event trigger times enters
    the neural network's input kernel. For each event and normalization
    period specified by `norm_seconds`, use time- shifted data from
    segments _before_ the event's segment tobuild up a background
    distribution of the output of matched filters of length `window_length`,
    normalized by the mean and standard deviation of the previous
    `norm_seconds` worth of data, until the effective time analyzed
    is equal to `max_tb`.
    The results of this analysis will be written to two csv files,
    one of which will contain the latency and false alaram rates
    for each of the events and normalization windows, and the other
    of which will contain the bins and counts for the background
    distributions used to calculate each of these false alarm rates.
    Args:
        data_dir: Path to directory containing timeslides and injections
        write_dir: Path to directory to which to write matched filter outputs
        results_dir:
            Path to directory to which to write analysis logs and
            summary csvs for analyzed events and their corresponding
            background distributions.
        window_length:
            Length of time, in seconds, over which to average
            neural network outputs for matched filter analysis
        t_clust: Clustering timescale for background distributions
        norm_seconds:
            Length of time, in seconds, over which to compute a moving
            "background" used to normalize the averaged neural network
            outputs. More specifically, the matched filter output at each
            point in time will be the average over the last `window_length`
            seconds, normalized by the mean and standard deviation of the
            previous `norm_seconds` seconds. If left as `None`, no
            normalization will be performed. Otherwise, should be specified
            as an iterable to compute multiple different normalization values
            for each event.
        max_tb:
            The maximum number of time-shifted background data to analyze
            per event, in seconds
        force:
            Flag indicating whether to force an event analysis to re-run
            if its data already exists in the summary files written to
            `results_dir`.
        log_file:
            A filename to write logs to. If left as `None`, logs will only
            be printed to stdout
        verbose:
            Flag indicating whether to log at level `INFO` (if set)
            or `DEBUG` (if not set)
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(results_dir / log_file, verbose)

    # initiate process and thread pools
    thread_ex = AsyncExecutor(4, thread=True)
    process_ex = AsyncExecutor(4, thread=False)

    # organize background and injection timeslides into segments
    background_segments = TimeSlide(
        data_dir / "dt-0.0-0.0", field="background-out"
    ).segments

    injection_segments = TimeSlide(
        data_dir / "dt-0.0-0.0", field="injection-out"
    ).segments

    with thread_ex, process_ex:
        # build background distributions
        # of all timeslides for various
        # normalization lengths
        with Progress() as pbar:
            backgrounds, sample_rate = build_background(
                thread_ex,
                process_ex,
                pbar,
                background_segments,
                data_dir,
                write_dir,
                max_tb,
                t_clust,
                window_length,
                norm_seconds,
            )

        # analyze all injection events
        # for all timeslides
        with Progress() as pbar:
            analyze_injections(
                thread_ex,
                process_ex,
                pbar,
                injection_segments,
                background_segments,
                data_dir,
                write_dir,
                results_dir,
                backgrounds,
                sample_rate,
            )


if __name__ == "__main__":
    main()
