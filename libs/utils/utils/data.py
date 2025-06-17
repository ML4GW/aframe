import logging
import math
import re
from pathlib import Path
from typing import List, Tuple


def segments_from_paths(paths: List[Path]):
    fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
    segments = []
    for fname in paths:
        match = fname_re.search(str(fname.path))
        if match is None:
            logging.warning(f"Couldn't parse file {fname.path}")

        start = float(match.group("t0"))
        duration = float(match.group("length"))
        stop = start + duration
        segments.append([start, stop])
    return segments


def get_num_shifts_from_Tb(
    segments: List[Tuple[float, float]],
    Tb: float,
    shift: float,
    psd_length: float,
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments and the desired total background duration.
    """
    # If Tb is zero, we want to do only zero-lag
    if Tb == 0:
        return 0
    livetime = 0
    num_shifts = 0

    # Subtract off time lost due to PSD burn-in
    durations = [stop - start - psd_length for start, stop in segments]
    # Increment the number of shifts we do until
    # enough background time is accumulated
    while livetime < Tb:
        num_shifts += 1
        for dur in durations:
            dur -= shift * num_shifts
            if dur > 0:
                livetime += dur
    return num_shifts


def get_num_shifts_from_num_signals(
    segments,
    num_signals: int,
    waveform_duration: float,
    spacing: float,
    shift: float,
    buffer: float,
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments, injection spacing, and the desired total
    number of injections
    """
    buffer += waveform_duration // 2
    spacing += waveform_duration
    T = sum([stop - start for start, stop in segments])
    a = -shift / 2
    b = T - 2 * buffer - (shift / 2)
    c = -num_signals * spacing
    discriminant = (b**2) - 4 * a * c
    N = (-b + (discriminant**0.5)) / (2 * a)
    return math.ceil(N)


def is_analyzeable_segment(
    start: float, stop: float, shifts: list[float], psd_length: float
) -> bool:
    """
    Given a segment start, stop, shift and psd length,
    validate if this segment is sufficiently long to be analyzed

    Args:
        start: start time of the segment
        stop: stop time of the segment
        shifts: list of shifts
        psd_length: length of the psd data
    """

    length = stop - start
    length -= max(shifts)
    length -= psd_length
    return length > 0
