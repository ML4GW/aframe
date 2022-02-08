import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from bbhnet import io
from bbhnet.analysis import matched_filter


def analyze_run(
    run_dir: Path,
    write_dir: Path,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    fnames: Optional[List[Path]] = None,
):
    os.makedirs(write_dir, exist_ok=True)

    # break data within run into contiguous segments
    groups = []
    this_group = []
    last_t0, length = None, None
    for fname in io.filter_and_sort_files(run_dir):
        if fnames is not None:
            run = Path(run_dir.parents[0].name)
            if run / "out" / fname not in  fnames:
                continue

        # don't need to check if match checkes
        # because we filtered those out in
        # "filter_and_sort_files"
        match = io.fname_re.search(fname)
        t0 = float(match.group("t0"))
        if last_t0 is not None and t0 != (last_t0 + length):
            # this file doesn't start where the last
            # one left off, so terminate the group
            # and start a new one
            groups.append(this_group)
            this_group = []

        last_t0 = t0
        length = float(match.group("length"))
        this_group.append(run_dir / fname)

    # add whatever group we were working on when
    # the loop terminated in to our group list
    groups.append(this_group)
    if len(groups[0]) == 0:
        raise ValueError(f"No groups in run directory {run_dir}")

    output_files = []
    for group in groups:
        t, y, mf = matched_filter.analyze_segment(group, norm_seconds)
        fname = io.write_data(write_dir, t, y, mf)
        output_files.append(fname)
    return output_files


@contextmanager
def impatient_pool(num_proc):
    try:
        ex = ProcessPoolExecutor(num_proc)
        yield ex
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def analyze_outputs_parallel(
    data_dir: Path,
    write_dir: Path,
    window_length: float = 1.0,
    num_proc: int = 1,
    shifts: Optional[List[float]] = None,
    fnames: Optional[List[Path]] = None,
    norm_seconds: Optional[float] = None,
):
    ex = ProcessPoolExecutor(num_proc)
    futures = []
    if fnames is not None:
        runs = list(set([f.parent.parent for f in fnames]))
    else:
        runs = None

    with impatient_pool(num_proc) as ex:
        shifts = shifts or os.listdir(data_dir)
        for shift in shifts:
            shift_dir = data_dir / shift
            _runs = runs or os.listdir(shift_dir)
            for run in _runs:
                future = ex.submit(
                    analyze_run,
                    shift_dir / run / "out",
                    os.path.join(write_dir, shift),
                    window_length,
                    norm_seconds,
                    fnames,
                )
                futures.append(future)

        for future in as_completed(futures):
            exc = future.exception()
            if isinstance(exc, FileNotFoundError):
                continue
            elif exc is not None:
                raise exc

            for output_file in future.result():
                yield output_file


def hhmmss(s):
    if s == 0:
        return "00:00:00.000"
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)

    m = str(int(m)).zfill(2)
    h = str(int(h)).zfill(2)
    return f"{h}:{m}:{s:2.3f}"


def build_background(
    data_dir: str,
    write_dir: str,
    num_bins: int,
    window_length: float = 1.0,
    num_proc: int = 1,
    fnames: Optional[List[Path]] = None,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
):
    min_mf, max_mf = None, None
    length = 0
    shifts = [i for i in os.listdir(data_dir) if i != "dt-0.0"]
    start_time = time.time()
    percent_completed = 0
    for fname in analyze_outputs_parallel(
        data_dir,
        write_dir,
        window_length=window_length,
        num_proc=num_proc,
        shifts=shifts,
        fnames=fnames,
        norm_seconds=norm_seconds,
    ):
        fnames.append(fname)

        minmax = io.minmax_re.search(fname)
        mn = float(minmax.group("min"))
        mx = float(minmax.group("max"))

        if min_mf is None:
            min_mf = mn
            max_mf = mx
        else:
            min_mf = min(min_mf, mn)
            max_mf = max(max_mf, mx)

        length += float(io.fname_re.search(fname).group("length"))
        logging.debug(f"Analyzed {length:0.1f}s of data")
        if max_tb is not None:
            if (length / max_tb) > (percent_completed + 0.01):
                percent_completed += 0.01
                elapsed = time.time() - start_time
                eta = (1 - percent_completed) * elapsed / percent_completed

                logging.info(
                    "Analyzed {}s of data, {}% complete in {}s, "
                    "estimated {}s remaining".format(
                        length,
                        int(percent_completed * 100),
                        hhmmss(elapsed),
                        hhmmss(eta),
                    )
                )

            if length >= max_tb:
                logging.info(
                    f"Analyzed {length:0.1f}s of data, terminating analysis"
                )
                break

    return fnames, length, min_mf, max_mf
