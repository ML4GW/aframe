import os
import re
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError,
    as_completed
)
from typing import List, Optional, Tuple


import h5py
import numpy as np
from scipy.signal import convolve


OUTPUT_DIR = "/home/tri.nguyen/gcp_data/hepcloud-output-jobsplit-full/"
WINDOW_LENGTH = 1
SAMPLE_RATE = 16  # TODO: get dynamically from diff between times
name_re = re.compile("out_(?P<t0>[0-9]{10})-(?P<length>[0-9]{2,5}).hdf5")


def read_fname(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(fname, "r") as f:
        t = f["GPSstart"][:] + 1
        y = f["out"][:, 0]
    return t, y


def match_filter(t, y, window_length: float = 1.0):
    sample_rate = 1 / (t[1] - t[0])

    window_size = int(window_length * sample_rate)
    window = np.ones((window_size,)) / window_size

    mf = convolve(y, window, mode="valid")
    t = t[window_size - 1:]  # slice off time samples to make y average of the past
    return t, mf


def analyze_segment(fnames: List[str], norm_seconds: Optional[int] = None):
    if norm_seconds is not None:
        total_seconds = sum(
            [int(name_re.search(f).group("length")) for f in fnames]
        )
        if total_seconds < (1.5 * norm_seconds):
            raise ValueError(
                "Segment from filenames {} has length {}s, "
                "analysis requires at least {}s of data".format(
                    fnames, total_seconds, 1.5 * norm_seconds
                )
            )

    t, y = zip(*map(read_fname, fnames))
    t = np.concatenate(t)
    y = np.concatenate(y)

    t, mf = match_filter(t, y)

    if norm_seconds is not None:
        _, shifts = match_filter(t, y, norm_seconds)
        _, sqs = match_filter(t, y ** 2, norm_seconds)
        scales = np.sqrt(sqs - shifts ** 2)

        mf = mf[-len(shifts):]
        mf = (mf - shifts) / scales

        t = t[-len(shifts):]
        y = y[-len(shifts):]
    else:
        y = y[-len(mf):]

    return t, y, mf


def search_run(run_dir: str, norm_seconds: Optional[float] = None):
    groups = []
    this_group = []
    last_t0 = None
    for fname in sorted(os.listdir(run_dir)):
        match = name_re.search(fname)
        if match is None:
            continue

        t0 = int(match.group("t0"))
        if last_t0 is None:
            last_t0 = t0
            length = int(match.group("length"))
        elif t0 != (last_t0 + length):
            groups.append(this_group)
            this_group = []
        last_t0 = t0

        length = int(match.group("length"))
        fname = os.path.join(run_dir, fname)
        this_group.append(fname)

    if len(this_group) > 0:
        groups.append(this_group)

    results = []
    for group in groups:
        t, y, snr = analyze_segment(group, norm_seconds)
        results.append((t, y, snr))
    return results



def parallel_search(
    num_proc: int,
    shifts: Optional[List[str]] = None,
    runs: Optional[List[int]] = None,
    norm_seconds: Optional[float] = None
):
    ex = ProcessPoolExecutor(num_proc)
    futures = []
    try:
        shifts = shifts or os.listdir(OUTPUT_DIR)
        runs = runs or range(17)
        for shift_dir in shifts:
            for run in map(str, runs):
                run_dir = os.path.join(OUTPUT_DIR, shift_dir, run, "out")
                future = ex.submit(search_run, run_dir, norm_seconds)
                futures.append(future)

        for future in as_completed(futures):
            exc = future.exception()
            if isinstance(exc, FileNotFoundError):
                continue
            elif exc is not None:
                raise exc

            for t, y, s in future.result():
                yield t, y, s
    finally:
        ex.shutdown(wait=False, cancel_futures=True)