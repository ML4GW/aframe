import logging
import re
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from analyze.utils import (
    AnalysisProgress,
    find_shift_and_foreground,
    load_segments,
)
from typeo import scriptify

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.distributions import ClusterDistribution
from bbhnet.analysis.normalizers import GaussianNormalizer
from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed

shift_pattern = re.compile(r"(?<=[HLKV])[0-9\.]+")


def fit_distribution(
    y: np.ndarray, t: np.ndarray, shifts: List, t_clust: float
) -> ClusterDistribution:
    dist = ClusterDistribution("integrated", ["H", "L"], t_clust)
    dist.fit((y, t), shifts)
    return dist


def get_write_field(distribution: str, norm: Optional[float]) -> str:
    field = f"{distribution}-integrated"
    if norm is not None:
        field += f"_norm-seconds={norm}"
    return field


def integrate_segments(
    pool: AsyncExecutor,
    write_dir: Path,
    yb: np.ndarray,
    yf: Optional[np.ndarray],
    t: np.ndarray,
    shift: str,
    t_clust: float,
    window_length: float,
    norm: Optional[float],
):
    shift_values = list(map(float, shift_pattern.findall(shift)))

    # build a normalizer for the given normalization window length
    norm = norm or None
    if norm is not None:
        sample_rate = 1 / (t[1] - t[0])
        normalizer = GaussianNormalizer(norm * sample_rate)
        normalizer.fit(yb)
    else:
        normalizer = None

    # integrate the nn outputs and use them
    # to fit a background distribution
    tb, yb, integrated = integrate(
        yb, t, window_length=window_length, normalizer=normalizer
    )

    # do the background fitting in parallel since
    # this will take the longest
    background_future = pool.submit(
        fit_distribution, integrated, tb, shift_values, t_clust
    )

    # write the integrated timeseries in this process
    field = get_write_field("background", norm)
    field_dir = write_dir / shift / field
    field_dir.mkdir(parents=True, exist_ok=True)
    write_timeseries(
        field_dir,
        prefix="integrated",
        t=tb,
        y=yb,
        integrated=integrated,
    )

    # repeat these steps for any foreground data
    if yf is not None:
        tf, yf, integrated = integrate(
            yf, t, window_length=window_length, normalizer=normalizer
        )
        foreground_future = pool.submit(
            fit_distribution, integrated, tf, shift_values, t_clust
        )
        field = get_write_field("foreground", norm)
        field_dir = write_dir / shift / field
        field_dir.mkdir(parents=True, exist_ok=True)
        write_timeseries(
            field_dir,
            prefix="integrated",
            t=tf,
            y=yf,
            integrated=integrated,
        )
    else:
        foreground_future = None

    return background_future, foreground_future


def aggregate_distributions(
    dist_type: str,
    futures: Dict[str, List[Future]],
    write_dir: Path,
    t_clust: float,
) -> None:
    def distribution_factory():
        return ClusterDistribution("integrated", ["H", "L"], t_clust)

    distributions = defaultdict(distribution_factory)
    for norm, mini_dist in as_completed(futures):
        distribution = distributions[norm]

        # update all the attributes of the global
        # distribution for this normalization value
        # with the attributes for one of the sub-
        # distributions fit on a single segment
        distribution.Tb += mini_dist.Tb
        distribution.events = np.append(distribution.events, mini_dist.events)
        distribution.event_times = np.append(
            distribution.event_times, mini_dist.event_times
        )
        distribution.shifts = np.append(
            distribution.shifts, mini_dist.shifts, axis=0
        )

    for norm, distribution in distributions.items():
        fname = dist_type
        if norm is not None:
            fname = fname + f"_norm-seconds={norm}"
        fname = fname + ".h5"
        distribution.write(write_dir / fname)


def fit_distributions(
    pool: AsyncExecutor,
    pbar: AnalysisProgress,
    background_segments: Iterable[Segment],
    foreground_field: str,
    data_dir: Path,
    write_dir: Path,
    max_tb: float,
    t_clust: float,
    window_length: float,
    norm_seconds: Optional[Iterable[float]] = None,
) -> None:
    norm_seconds = norm_seconds or [norm_seconds]
    fit_futures = {
        "foreground": defaultdict(list),
        "background": defaultdict(list),
    }

    background_segments = iter(background_segments)
    while pbar.Tb < max_tb:
        try:
            segment = next(background_segments)
        except StopIteration:
            break

        # iterate through all possible timeshifts of this zero-shifted
        # segment and load in both the background data as well as
        # any foreground injection data if it exists
        analysis_jobs, load_futures = 0, {}
        for shift in segment.root.parent.parent.iterdir():
            if not shift.is_dir():
                continue

            background, foreground = find_shift_and_foreground(
                segment, shift.name, foreground_field
            )
            if background is None:
                continue

            future = pool.submit(load_segments, (background, foreground))
            load_futures[shift.name] = [future]
            analysis_jobs += 1 if foreground is None else 2

        # create progress bar tasks for each one
        # of the subprocesses involved for analyzing
        # this set of timeslide ClusterDistributions
        load_task_id, analyze_task_id, write_task_id = pbar.get_tasks(
            len(load_futures),
            analysis_jobs * len(norm_seconds),
            segment.length,
        )

        load_cb = pbar.get_task_cb(load_task_id)
        analyze_cb = pbar.get_task_cb(analyze_task_id)

        for f in load_futures.values():
            f[0].add_done_callback(load_cb)

        # as these loads complete, integrate and normalize the
        # background and foreground timeseries using each of
        # the specified normalization values and fit distributions
        # to the integrated values
        for shift, (yf, yb, t) in as_completed(load_futures):
            for norm in norm_seconds:
                background_future, foreground_future = integrate_segments(
                    pool,
                    write_dir=write_dir,
                    yb=yb,
                    yf=yf,
                    t=t,
                    shift=shift,
                    t_clust=t_clust,
                    window_length=window_length,
                    norm=norm,
                )
                background_future.add_done_callback(analyze_cb)
                fit_futures["background"][norm].append(background_future)
                pbar.update(write_task_id, advance=1)

                if foreground_future is not None:
                    foreground_future.add_done_callback(analyze_cb)
                    fit_futures["foreground"][norm].append(foreground_future)
                    pbar.update(write_task_id, advance=1)

            advance = t[-1] - t[0] + t[1] - t[0]
            pbar.update(pbar.main_task_id, advance=advance)

    logging.info(f"Accumulated {pbar.Tb}s of background")
    for dist_type, futures in fit_futures.items():
        aggregate_distributions(dist_type, futures, write_dir, t_clust)


@scriptify
def main(
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    t_clust: float,
    window_length: Optional[float] = None,
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
    configure_logging(log_file, verbose)

    # organize background and injection timeslides into segments
    zero_shift = data_dir / "dt-H0.0-L0.0"
    background_ts = TimeSlide(zero_shift, field="background-out")
    background_segments = background_ts.segments

    pool = AsyncExecutor(4, thread=False)
    pbar = AnalysisProgress(max_tb)
    with pool, pbar:
        fit_distributions(
            pool,
            pbar,
            background_segments,
            foreground_field="injection-out",
            data_dir=data_dir,
            write_dir=write_dir,
            max_tb=max_tb,
            t_clust=t_clust,
            window_length=window_length,
            norm_seconds=norm_seconds,
        )


if __name__ == "__main__":
    main()
