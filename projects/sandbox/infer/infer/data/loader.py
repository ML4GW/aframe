import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty, Full
from typing import List, Optional

import h5py
import numpy as np


def load_fname(
    fname: Path, channels: List[str], shifts: List[int], chunk_size: int
) -> np.ndarray:
    """
    Yield chunks of time-shifted data from a background file. Note that
    `shifts` is specified in number of indices for this function, as opposed
    to being in seconds as it is elsewhere.
    """
    max_shift = max(shifts)
    with h5py.File(fname, "r") as f:
        size = len(f[channels[0]]) - max_shift
        idx = 0
        while idx < size:
            data = []
            for channel, shift in zip(channels, shifts):
                start = idx + shift
                stop = start + chunk_size

                # make sure that segments with shifts shorter
                # than the max shift get their ends cut off
                stop = min(size + shift, stop)
                x = f[channel][start:stop]
                data.append(x)

            yield np.stack(data).astype("float32")
            idx += chunk_size


def crawl_through_directory(
    data_dir: Path,
    channels: List[str],
    chunk_length: float,
    sample_rate: float,
    shifts: Optional[List[float]],
):
    """
    Go through a directory, finding all files that match the pattern
    of a prefix, followed by `<GPS timestamp>-<duration>`, followed
    by a suffix. Yield the data in those files in chunks, as well
    as the starting and ending time of the data from each file.
    """
    fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
    chunk_size = int(chunk_length * sample_rate)

    if shifts is not None:
        max_shift = max(shifts)
        shifts = [int(i * sample_rate) for i in shifts]
    else:
        max_shift = 0
        shifts = [0 for _ in channels]

    for fname in data_dir.iterdir():
        match = fname_re.search(fname.name)
        if match is None:
            continue

        # first return some information about
        # the segment we're about to iterate through
        start = float(match.group("t0"))
        duration = float(match.group("length"))
        yield (start, start + duration - max_shift)

        # now iterate through the segment in chunks
        for x in load_fname(fname, channels, shifts, chunk_size):
            yield x

        # now return None to indicate this segment is done
        yield None

    # finally a back-to-back None to indicate
    # that all segments are completed
    yield None


@dataclass
class ChunkedSegmentLoader:
    """
    Iterable context which, upon entering, launches chunked loading
    of the files in `data_dir` in an asychronous process that gets closed
    upon context exit. The object returned by `__enter__` is a iterator that,
    for each file in `data_dir`, returns a generator that iterates through that
    file's content (possibly with relative shifts between channels) in chunks
    of length `chunk_length`.

    Example usage:
    ```python
    data_dir = Path("/path/to/data")
    loader = ChunkedSegmentLoader(
        data_dir,
        channels=["H1", "L1"],
        chunk_length=1024,
        sample_rate=2048,
        shifts=[0, 1]
    )
    with loader:
        # async loading has launched
        for iterator in loader:
            # each file in data_dir has its own iterator
            for chunk in iterator:
                # note that this won't be true in general for the
                # last chunk in each file, which might be shorter
                # than the specified chunk length
                assert chunk.shape == (2, 2048 * 1024)
    ```
    """

    data_dir: Path
    channels: List[str]
    chunk_length: float
    sample_rate: float
    shifts: Optional[List[float]]

    def __enter__(self):
        self.q = Queue(1)
        self.event = Event()
        self.done_event = Event()
        self.clear_event = Event()
        self.p = Process(target=self)
        self.p.start()
        return self._iter_through_q()

    def __exit__(self, *_):
        # set the event to let the child process
        # know that we're done with whatever data
        # it's generating and it should stop
        self.event.set()

        # wait for the child to indicate to us
        # that it has been able to finish gracefully
        while not self.done_event.is_set():
            time.sleep(1e-3)

        # remove any remaining data from the queue
        # to flush the child process's buffer so
        # that it can exit gracefully, then close
        # the queue from our end
        self._clear_q()
        self.q.close()
        self.clear_event.set()

        # now wait for the child to exit
        # gracefully then close it
        self.p.join()
        self.p.close()

    def __call__(self):
        try:
            it = crawl_through_directory(
                self.data_dir,
                self.channels,
                self.chunk_length,
                self.sample_rate,
                self.shifts,
            )
            while not self.event.is_set():
                x = next(it)
                self.try_put(x)
        except Exception:
            exc_type, exc, tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc, tb)
            tb = "".join(tb[:-1])
            self.try_put((exc_type, str(exc), tb))
        finally:
            # now let the parent process know that
            # there's no more information going into
            # the queue, and it's free to empty it
            self.done_event.set()

            # if we arrived here from an exception, i.e.
            # the event has not been set, then don't
            # close the queue until the parent process
            # has received the error message and set the
            # event itself, otherwise it will never be
            # able to receive the message from the queue
            while not self.event.is_set() or not self.clear_event.is_set():
                time.sleep(1e-3)

            self.q.close()
            self.q.cancel_join_thread()

    def try_put(self, x):
        while not self.event.is_set():
            try:
                self.q.put_nowait(x)
            except Full:
                time.sleep(1e-3)
            else:
                break

    def try_get(self):
        while not self.event.is_set():
            try:
                x = self.q.get_nowait()
            except Empty:
                time.sleep(1e-3)
                continue

            if isinstance(x, tuple) and len(x) == 3:
                exc_type, msg, tb = x
                logging.exception(
                    "Encountered exception in data collection process:\n" + tb
                )
                raise exc_type(msg)
            return x

    def _clear_q(self):
        while True:
            try:
                self.q.get_nowait()
            except Empty:
                break

    def segment_gen(self):
        while True:
            x = self.try_get()
            if x is None:
                break
            yield x

    def _iter_through_q(self):
        while True:
            x = self.try_get()
            if x is None:
                break
            start, stop = x
            yield (start, stop), self.segment_gen()
