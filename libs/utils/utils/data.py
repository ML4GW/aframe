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


def calc_shifts_required(Tb: float, T: float, delta: float) -> int:
    r"""
    Calculate the number of shifts required to generate Tb
    seconds of background.

    Solve:
    $$\sum_{i=1}^{N}(T - i\delta) \geq T_b$$
    for the lowest value of N, where \delta is the
    shift increment.

    TODO: generalize to multiple ifos and negative
    shifts, since e.g. you can in theory get the same
    amount of Tb with fewer shifts if for each shift
    you do its positive and negative. This should just
    amount to adding a factor of 2 * number of ifo
    combinations in front of the sum above.
    """

    discriminant = (delta / 2 - T) ** 2 - 2 * delta * Tb
    N = (T - delta / 2 - discriminant**0.5) / delta
    return math.ceil(N)


def get_num_shifts_from_Tb(
    segments: List[Tuple[float, float]], Tb: float, shift: float
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments and the desired total background duration.
    """
    T = sum([stop - start for start, stop in segments])
    return calc_shifts_required(Tb, T, shift)


def get_num_shifts_from_num_injections(
    segments,
    num_injections: int,
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
    c = -num_injections * spacing
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
