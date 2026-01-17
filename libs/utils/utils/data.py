import logging
import math
import re
from pathlib import Path


def segments_from_paths(paths: list[Path]) -> list[list[float]]:
    """
    Extract segment start and stop times from filenames.

    Parses filenames with the format 't0-length' where t0 is the
    GPS start time and length is the duration in seconds. Returns
    a list of [start, stop] time pairs for each segment.

    Args:
        paths (list[Path]): List of Path objects containing filenames
            to parse. The filenames are expected to contain the following
            format: 't0-length', where t0 is the GPS start time (can be
            a float) and length is the duration in seconds (can also be
            a float).

    Returns:
        list[list[float]]: List of [start_time, stop_time] pairs where times
            are in seconds.

    Raises:
        logging.warning: If a filename cannot be parsed with the expected
            format, a warning is logged but processing continues.

    Examples:
        >>> segments_from_paths([Path('background-1234567890-1000.hdf5')])
        [[1234567890.0, 1234568890.0]]

        >>> segments_from_paths([Path('background-1000000000.5-1000.25.hdf5')])
        [[1000000000.5, 1000001000.75]]
    """
    # Regex pattern to match 't0-length' format where both can be decimals
    fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
    segments = []

    for fname in paths:
        match = fname_re.search(str(fname.path))
        if match is None:
            logging.warning(f"Couldn't parse file {fname.path}")
            continue

        # Extract start time and duration from regex groups
        start = float(match.group("t0"))
        duration = float(match.group("length"))
        stop = start + duration
        segments.append([start, stop])

    return segments


def get_num_shifts_from_Tb(
    segments: list[tuple[float, float]],
    Tb: float,
    shift: float,
    psd_length: float,
) -> int:
    """
    Calculate the number of required time shifts to accumulate target duration.

    Computes how many time shifts are needed to accumulate at least Tb seconds
    of background data, accounting for PSD burn-in time and the shift amount.

    Args:
        segments (list[tuple[float, float]]): List of (start_time, stop_time)
            tuples defining available segments.
        Tb (float): Target background duration in seconds. If 0, returns 0
            (zero-lag only).
        shift (float): Maximum time shift in seconds for each
            consecutive shift.
        psd_length (float): Length of PSD data in seconds that is unusable
            for background.

    Returns:
        int: Number of time shifts needed to accumulate at least Tb seconds
            of background livetime.

    Examples:
        >>> segments = [(0, 10000), (20000, 30000)]
        >>> get_num_shifts_from_Tb(segments, 86400, 1, 64)
        5
    """
    # Special case: zero-lag only analysis
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
    segments: list[tuple[float, float]],
    num_signals: int,
    waveform_duration: float,
    spacing: float,
    shift: float,
    buffer: float,
) -> int:
    """
    Calculate number of time shifts needed to inject desired number of signals.

    Uses the quadratic equation to determine the minimum number of shifts
    required to fit the desired number of injections with given
    spacing constraints.

    Args:
        segments (list[tuple[float, float]]): List of (start_time, stop_time)
            tuples defining available segments.
        num_signals (int): Desired total number of signal injections across
            all shifts.
        waveform_duration (float): Duration of each waveform in seconds.
        spacing (float): Minimum spacing between injections in seconds.
        shift (float): Maximum time shift in seconds for each
            consecutive shift.
        buffer (float): Buffer at segment edges in seconds where injections
            cannot occur.

    Returns:
        int: Minimum number of time shifts needed to accommodate num_signals
            injections.

    Examples:
        >>> segments = [(0, 10000), (20000, 30000)]
        >>> get_num_shifts_from_num_signals(
        ...     segments, 2000, 8, 16, 1, 8
        ... )
        3
    """
    # Adjust buffer and spacing to account for waveform duration
    buffer += waveform_duration // 2
    spacing += waveform_duration

    # Calculate total available time across all segments
    T = sum([stop - start for start, stop in segments])

    # Quadratic equation coefficients: a*N^2 + b*N + c = 0
    # where N is the number of shifts needed
    a = -shift / 2
    b = T - 2 * buffer - (shift / 2)
    c = -num_signals * spacing
    discriminant = (b**2) - 4 * a * c
    N = (-b + (discriminant**0.5)) / (2 * a)

    return math.ceil(N)


def is_analyzeable_segment(
    start: float,
    stop: float,
    shifts: list[float],
    psd_length: float,
) -> bool:
    """
    Validate whether a segment has sufficient duration for analysis.

    Checks if a segment remains analyzeable after accounting for
    PSD burn-in time and the maximum time shift required.
    A segment is analyzeable if: (stop - start) - max(shifts) - psd_length > 0

    Args:
        start (float): Start time of the segment in seconds.
        stop (float): Stop time of the segment in seconds.
        shifts (list[float]): List of time shift values in seconds. The maximum
            value is used to determine time loss from shifting.
        psd_length (float): Length of PSD data in seconds that is unusable
            for analysis.

    Returns:
        bool: True if segment has positive duration after accounting for
            PSD length and maximum shift, False otherwise.

    Examples:
        >>> is_analyzeable_segment(0, 1000, [0, 1, 2], 64)
        True

        >>> is_analyzeable_segment(0, 65, [0, 1, 2], 64)
        False
    """
    # Calculate total segment duration
    length = stop - start

    # Subtract time lost to maximum shift
    length -= max(shifts)

    # Subtract PSD burn-in time
    length -= psd_length

    return length > 0
