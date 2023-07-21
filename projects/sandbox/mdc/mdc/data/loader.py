import logging
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty, Full
from typing import List

import h5py
import numpy as np


def aggregate(data):
    return np.stack(data).astype("float32")


def load_fname(
    data_dir: Path, dataset: str, channels: List[str], chunk_size: int
) -> np.ndarray:
    bg_fname = data_dir / "background.hdf"
    fg_fname = data_dir / "foreground.hdf"

    bg_file = h5py.File(bg_fname, "r")
    fg_file = h5py.File(fg_fname, "r")
    with bg_file as bgf, fg_file as fgf:
        size = len(bgf[channels[0]][dataset])
        idx = 0
        while idx < size:
            bg, fg = [], []
            for channel in channels:
                start = idx
                stop = start + chunk_size

                # make sure that segments with shifts shorter
                # than the max shift get their ends cut off
                bgx = bgf[channel][dataset][start:stop]
                fgx = fgf[channel][dataset][start:stop]

                bg.append(bgx)
                fg.append(fgx)

            yield [aggregate(x) for x in [bg, fg]]
            idx += chunk_size


@dataclass
class ChunkedSegmentLoader:
    data_dir: Path
    dataset: str
    channels: List[str]
    chunk_size: float

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
            it = load_fname(
                self.data_dir, self.dataset, self.channels, self.chunk_size
            )
            while not self.event.is_set():
                try:
                    x = next(it)
                except StopIteration:
                    self.try_put(None)
                else:
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

    def _iter_through_q(self):
        while True:
            x = self.try_get()
            if x is None:
                break
            yield x
