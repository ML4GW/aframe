import logging
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Tuple


def read_stream(stream, process, q):
    stream = getattr(process, stream)
    try:
        it = iter(stream.readline, b"")
        while True:
            try:
                line = next(it)
            except StopIteration:
                break
            q.put(line.decode())
    finally:
        q.put(None)


def stream_process(process):
    q = Queue()
    args = (process, q)
    streams = ["stdout", "stderr"]
    threads = [Thread(target=read_stream, args=(i,) + args) for i in streams]
    for t in threads:
        t.start()

    for _ in range(2):
        for line in iter(q.get, None):
            sys.stdout.write(line)


def stream_command(command: List[str]):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ
    )
    stream_process(process)

    process.wait()
    if process.returncode:
        raise RuntimeError(
            "Command '{}' failed with return code {} "
            "and stderr:\n{}".format(
                shlex.join(command),
                process.returncode,
                process.stderr.read().decode(),
            )
        ) from None


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
    spacing: float,
    shift: float,
    buffer: float,
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments, injection spacing, and the desired total
    number of injections
    """
    T = sum([stop - start for start, stop in segments])
    a = -shift / 2
    b = T - 2 * buffer - (shift / 2)
    c = -num_injections * spacing
    discriminant = b**2 - 4 * a * c
    N = math.ceil(-b + discriminant**0.5) / (2 * a)
    return N
