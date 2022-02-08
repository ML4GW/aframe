import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List, Optional

from bbhnet import io
from bbhnet.analysis import matched_filter


def analyze_run(
    run_dir: str,
    write_dir: str,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
):
    os.makedirs(write_dir, exist_ok=True)

    # break data within run into contiguous segments
    groups = []
    this_group = []
    last_t0, length = None, None
    for fname in io.filter_and_sort_files(os.listdir(run_dir)):
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

        fname = os.path.join(run_dir, fname)
        this_group.append(fname)

    # add whatever group we were working on when
    # the loop terminated in to our group list
    groups.append(this_group)

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


def _run_in_range(
    run_dir: str, t0: Optional[float], length: Optional[float]
) -> bool:
    if t0 is None:
        return True

    fnames = io.filter_and_sort_files(os.listdir(run_dir))
    t0s = [float(io.fname_re.search(f).group("t0")) for f in fnames]
    if all([t0 > t for t in t0s]):
        return False
    elif length is not None and all([(t0 + length) < t for t in t0s]):
        return False
    return True


def analyze_outputs_parallel(
    data_dir: str,
    write_dir: str,
    window_length: float = 1.0,
    num_proc: int = 1,
    shifts: Optional[List[float]] = None,
    t0: Optional[float] = None,
    length: Optional[float] = None,
    norm_seconds: Optional[float] = None,
):
    ex = ProcessPoolExecutor(num_proc)
    futures = []

    with impatient_pool(num_proc) as ex:
        shifts = shifts or os.listdir(data_dir)
        for shift in shifts:
            shift_dir = os.path.join(data_dir, shift)
            for run in os.listdir(shift_dir):
                run_dir = os.path.join(shift_dir, run, "out")
                in_range = _run_in_range(run_dir, t0, length)
                if not in_range:
                    continue

                future = ex.submit(
                    analyze_run,
                    run_dir,
                    os.path.join(write_dir, shift),
                    window_length,
                    norm_seconds,
                    t0,
                    length
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
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)

    m = str(int(m)).zfill(2)
    h = str(int(h)).zfill(2)
    return f"{h}:{m}:{s:0.3f}"


def build_background(
    data_dir: str,
    write_dir: str,
    num_bins: int,
    window_length: float = 1.0,
    num_proc: int = 1,
    t0: Optional[float] = None,
    length: Optional[float] = None,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
):
    min_mf, max_mf = None, None
    length = 0
    shifts = [i for i in os.listdir(data_dir) if i != "dt-0.0"]
    fnames = []
    start_time = time.time()
    percent_completed = 0
    for fname in analyze_outputs_parallel(
        data_dir,
        write_dir,
        window_length=window_length,
        num_proc=num_proc,
        shifts=shifts,
        t0=t0,
        length=length,
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
