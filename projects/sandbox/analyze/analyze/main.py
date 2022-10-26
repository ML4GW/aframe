import logging
from concurrent.futures import FIRST_EXCEPTION, wait
from pathlib import Path
from typing import Iterable, List, Optional

from analyze.utils import (
    CallbackFactory,
    find_shift_and_foreground,
    load_segments,
)

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.normalizers import GaussianNormalizer
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed
from hermes.typeo import typeo


def fit_distributions(
    pool: AsyncExecutor,
    pbar: CallbackFactory,
    background_segments: Iterable[Segment],
    foreground_field: str,
    data_dir: Path,
    max_tb: float,
    window_length: float,
    norm_seconds: Optional[Iterable[float]] = None,
):
    norm_seconds = norm_seconds or [norm_seconds]
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
        load_cb, analyze_cb, write_cb = pbar.get_task_cbs(
            len(load_futures),
            analysis_jobs * len(norm_seconds),
            segment.length,
        )
        for f in load_futures.values():
            f[0].add_done_callback(load_cb)

        # as these loads complete, integrate and normalize the
        # background and foreground timeseries using each of
        # the specified normalization values and fit distributions
        # to the integrated values
        for shift, (yf, yb, t) in as_completed(load_futures):
            sample_rate = 1 / (t[1] - t[0])
            for norm in norm_seconds:
                # treat 0 the same as no normalization
                norm = norm or None

                # build a normalizer for the given normalization window length
                if norm is not None:
                    normalizer = GaussianNormalizer(norm * sample_rate)
                    normalizer.fit(yb)
                else:
                    normalizer = None

                # integrate the nn outputs and use them to fit
                # a background distribution. The `distributions`
                # dict will create a new distribution if one doesn't
                # already exist for this normalization value
                future = pool.submit(
                    integrate,
                    yb,
                    t,
                    window_length=window_length,
                    normalizer=normalizer,
                )
                cb = pbar.get_cb("background", norm, shift, write_cb)
                future.add_done_callback(cb)
                future.add_done_callback(analyze_cb)

                if yf is not None:
                    future = pool.submit(
                        integrate,
                        yf,
                        t,
                        window_length=window_length,
                        normalizer=normalizer,
                    )
                    cb = pbar.get_cb("foreground", norm, shift, write_cb)
                    future.add_done_callback(cb)
                    future.add_done_callback(analyze_cb)

            advance = t[-1] - t[0] + t[1] - t[0]
            pbar.update(pbar.main_task_id, advance=advance)

    logging.info(f"Accumulated {pbar.Tb}s of background")
    pbar.write_distributions()
    return pbar.distributions, pbar.write_futures


@typeo
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
    pbar = CallbackFactory(
        ["H1", "L1"],
        t_clust=t_clust,
        max_tb=max_tb,
        write_dir=write_dir,
        pool=pool,
    )
    with pool, pbar:
        backgrounds, write_futures = fit_distributions(
            pool,
            pbar,
            background_segments,
            foreground_field="injection-out",
            data_dir=data_dir,
            max_tb=max_tb,
            window_length=window_length,
            norm_seconds=norm_seconds,
        )
        wait(write_futures, return_when=FIRST_EXCEPTION)


if __name__ == "__main__":
    main()
