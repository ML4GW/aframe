import logging
import random
from typing import List
from zlib import adler32

import numpy as np


def seed_worker(start: float, stop: float, shifts: List[float], seed: int):
    fingerprint = str((start, stop) + tuple(shifts))
    worker_hash = adler32(fingerprint.encode())
    logging.info(
        "Seeding data generation with seed {}, "
        "augmented by worker seed {}".format(seed, worker_hash)
    )
    np.random.seed(seed + worker_hash)
    random.seed(seed + worker_hash)


def calc_segment_injection_times(
    start: float,
    stop: float,
    spacing: float,
    buffer: float,
    waveform_duration: float,
):
    """
    Calculate the times at which to inject signals into a segment

    Args:
        start:
            The start time of the segment
        stop:
            The stop time of the segment
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
        waveform_duration:
            The duration of the waveform in seconds

    Returns: np.ndarray of injection times
    """

    buffer += waveform_duration // 2
    spacing += waveform_duration
    injection_times = np.arange(start + buffer, stop - buffer, spacing)
    return injection_times
