import logging
import math
import re
from typing import List, Tuple

from law.target.base import Target


def segments_from_paths(paths: List[Target]):
    fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
    segments = []
    for fname in paths:
        match = fname_re.search(fname.path)
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

    The algebra to get this is gross but straightforward.
    Just solving
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


def get_num_shifts(
    segments: List[Tuple[float, float]], Tb: float, shift: float
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments
    """
    T = sum([stop - start for start, stop in segments])
    return calc_shifts_required(Tb, T, shift)
