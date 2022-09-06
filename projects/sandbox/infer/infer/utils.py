import logging
import time
from concurrent.futures import wait
from itertools import chain, product
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

import numpy as np

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.parallelize import AsyncExecutor


def load(
    segment: Segment,
    write_dir: Path,
    stride_size: int,
    batch_size: int,
    fduration: Optional[float] = None,
):
    hanford, t = segment.load("H1")
    livingston, _ = segment.load("L1")
    logging.info(f"Loaded data from segment {segment}")

    stream_size = stride_size * batch_size
    num_streams = len(t) // stream_size
    t = t[: num_streams * stream_size : stride_size]
    if fduration is not None:
        t -= fduration / 2

    x = np.stack([hanford, livingston])
    x = x[:, : num_streams * stream_size].astype("float32")
    x = np.split(x, num_streams, axis=-1)

    write_dir = write_dir / segment.shift_dir / f"{segment.field}-out"
    sequence = Sequence(t, x, write_dir, batch_size)
    return sequence


class Sequence:
    def __init__(self, t, x, write_dir, batch_size):
        self.t = t
        self.x = x
        self.y = np.zeros_like(t)
        self.batch_size = batch_size
        self.write_dir = write_dir

        self._last_seen = -1

    def update(self, y, request_id, sequence_id):
        if request_id > (self._last_seen + 1):
            logging.warning(
                "Dropped response for request {} for sequence {}".format(
                    self._last_seen + 1, sequence_id
                )
            )
            self._last_seen = request_id
        elif request_id < self._last_seen:
            logging.warning(
                "Request {} from sequence {} came in late".format(
                    request_id, sequence_id
                )
            )
        else:
            self._last_seen = request_id

        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self.y[start:stop] = y[:, 0]

    @property
    def finished(self):
        return self._last_seen == (len(self.x) - 1)


class SequenceManager:
    def __init__(
        self,
        data_dir: Path,
        write_dir: Path,
        fields: Iterable[str],
        client,
        stride_size: int,
        batch_size: int,
        inference_rate: float,
        num_io_workers: int,
        max_streams: int,
        max_seconds: float,
        fduration: Optional[float] = None,
        base_sequence_id: int = 1001,
    ) -> None:
        self.client = client
        self.client.callback = self.callback

        self.write_dir = write_dir
        self.stride_size = stride_size
        self.batch_size = batch_size
        self.fduration = fduration
        self.inference_rate = inference_rate
        self.base_sequence_id = base_sequence_id

        # do reading and writing of segments using multiprocessing
        self.io_pool = AsyncExecutor(num_io_workers, thread=False)

        # submit inference requests using threads so that we
        # can leverage all of our snapshotter states on the server
        self.infer_pool = AsyncExecutor(max_streams, thread=True)

        self.max_streams = max_streams
        self.max_seconds = max_seconds
        self._num_seconds = 0

        self._timeslide_it = product(data_dir.iterdir(), fields)
        self._segment_it = None

        self.lock = Lock()
        self._done = False
        self.futures = []
        self.sequences = {}

    def is_done(self):
        with self.lock:
            done, not_done = wait(self.futures, timeout=1e-6)
            for f in list(done):
                exc = f.exception()
                if exc is not None:
                    raise exc

            self.futures = list(not_done)
            return len(self.futures) == 0 and self._done

    def add(self, sequence: Sequence) -> int:
        with self.lock:
            for i in range(len(self.sequences) + 1):
                seq_id = self.base_sequence_id + i
                if seq_id not in self.sequences:
                    break
            self.sequences[seq_id] = sequence

        logging.info(
            "Managing sequence with t0={} as id {}".format(
                sequence.t[0], seq_id
            )
        )
        return seq_id

    def infer(self, sequence: Sequence):
        sequence_id = self.add(sequence)
        logging.debug(f"Beginning inference on sequence {sequence_id}")
        for i, update in enumerate(sequence.x):
            logging.debug(
                "Submitting request {} for sequence {}".format(i, sequence_id)
            )
            try:
                self.client.infer(
                    update,
                    request_id=i,
                    sequence_id=sequence_id,
                    sequence_start=i == 0,
                    sequence_end=i == (len(sequence.x) - 1),
                )
            except Exception as e:
                logging.error(str(e))
                raise e
            time.sleep(1 / self.inference_rate)

            # let the first request complete before we
            # send anymore to make sure that the request
            # with sequence_start=True gets there first
            if i == 0:
                while sequence._last_seen < 0:
                    time.sleep(1e-3)

    def callback(self, y, request_id, sequence_id):
        with self.lock:
            self.sequences[sequence_id].update(y, request_id, sequence_id)
            if self.sequences[sequence_id].finished:
                logging.info(f"Finished inference on sequence {sequence_id}")
                sequence = self.sequences.pop(sequence_id)
                self._num_seconds -= sequence.t[-1] - sequence.t[0]

                write_future = self.io_pool.submit(
                    write_timeseries,
                    sequence.write_dir,
                    y=sequence.y,
                    t=sequence.t,
                )
                self.futures.append(write_future)

                segment = self.get_next_segment()
                if segment is not None:
                    load_future = self.submit_load(segment)
                    self.futures.append(load_future)

                return write_future

    def start_new_timeslide(self, root: Path, field: str):
        write_dir = self.write_dir / root.name / f"{field}-out"
        write_dir.mkdir(parents=True, exist_ok=True)
        timeslide = TimeSlide(root, field)
        self._segment_it = iter(timeslide.segments)

    def get_next_segment(self):
        if self._done or len(self.sequences) >= self.max_streams:
            return

        try:
            segment = next(self._segment_it)
        except (StopIteration, TypeError):
            # either _segment_it is None or it's exhausted,
            # so initialize a new timeslide and start iterating
            # through its segments
            try:
                root, field = next(self._timeslide_it)
            except StopIteration:
                self._done = True
                return

            self.start_new_timeslide(root, field)
            return self.get_next_segment()

        if (self._num_seconds + segment.length) > self.max_seconds:
            self._segment_it = chain([segment], self._segment_it)
            return

        self._num_seconds += segment.length
        return segment

    def load_callback(self, future):
        infer_future = self.infer_pool.submit(self.infer, future.result())
        with self.lock:
            self.futures.append(infer_future)

    def submit_load(self, segment: Segment):
        future = self.io_pool.submit(
            load,
            segment,
            self.write_dir,
            self.stride_size,
            self.batch_size,
            self.fduration,
        )
        future.add_done_callback(self.load_callback)
        return future

    def __enter__(self):
        self.io_pool.__enter__()
        self.infer_pool.__enter__()

        for _ in range(self.max_streams):
            with self.lock:
                segment = self.get_next_segment()

            if segment is None:
                break

            future = self.submit_load(segment)
            self.futures.append(future)

    def __exit__(self, *exc_args):
        self.io_pool.__exit__(*exc_args)
        self.infer_pool.__exit__(*exc_args)
