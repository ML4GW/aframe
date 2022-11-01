import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

t0_pattern = re.compile(r"[0-9]{10}(\.[0-9])?(?=-)")
dur_pattern = re.compile("[0-9]{2,8}(?=.hdf5)")


def dirname_to_shifts(dirname: str):
    shifts = dirname.split("-")[1:]
    shifts = [float(i[1:]) for i in shifts]
    return np.array(shifts)


def get_strain_dirname(event_type: str) -> str:
    if event_type == "foreground":
        return "injection"
    else:
        return "background"


def get_response_dirname(event_type: str, norm: Optional[float] = None) -> str:
    int_dirname = f"{event_type}-integrated"
    if norm is not None and float(norm) > 0:
        return int_dirname + f"_norm-seconds={float(norm)}"
    return int_dirname


def get_event_fname(data_dir: Path, event_time: float):
    for fname in data_dir.iterdir():
        try:
            t0 = float(t0_pattern.search(fname.name).group(0))
            dur = float(dur_pattern.search(fname.name).group(0))
        except AttributeError:
            continue

        if t0 <= event_time < (t0 + dur):
            break
    else:
        raise ValueError(
            "No file containing event time {} in "
            "data directory {}".format(event_time, data_dir)
        )

    return fname


def get_segments(data_dir: Path) -> List[Tuple[float, float]]:
    segments = []
    for fname in data_dir.iterdir():
        try:
            t0 = float(t0_pattern.search(fname.name).group(0))
            dur = float(dur_pattern.search(fname.name).group(0))
        except AttributeError:
            continue

        segments.append((t0, t0 + dur))

    if len(segments) == 0:
        raise ValueError(
            f"No properly formatted segments in data directory {data_dir}"
        )

    return segments
