import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.typeo import typeo


def load(segment: Segment):
    hanford, t = segment.load("H1")
    livingston, _ = segment.load("L1")
    return np.stack([hanford, livingston]), t


def stream_data(
    x: np.ndarray, stream_size: int, sequence_id: int, client: InferenceClient
):

    # TODO: added this b/c there was a complaint
    # should the required data types be an argument
    # to the h5.write functions?
    x = x.astype(np.float32)
    num_streams = (x.shape[-1] - 1) // stream_size + 1
    for i in range(num_streams):
        stream = x[:, i * stream_size : (i + 1) * stream_size]
        client.infer(
            stream,
            request_id=i,
            sequence_id=sequence_id,
            sequence_start=i == 0,
            sequence_end=(i + 1) == num_streams,
        )
        time.sleep(1e-3)


def infer(
    client: InferenceClient,
    executor: AsyncExecutor,
    timeslides: Iterable[TimeSlide],
    write_dir: Path,
    stream_size: int,
    base_sequence_id: int,
):
    for timeslide in timeslides:

        ts_write_dir = write_dir / timeslide.shift / f"{timeslide.field}-out"
        ts_write_dir.mkdir(parents=True, exist_ok=True)
        timeseries = {}

        data_it = executor.imap(load, timeslide.segments)
        for i, (x, t) in enumerate(data_it):
            sequence_id = base_sequence_id + i

            # create a new timeseries entry to keep track of
            # as results come back in from the inference server
            t = t[::stream_size]
            timeseries[sequence_id] = {"t": t, "y": np.array([])}

            # submit all the streaming inference requests to
            # the inference server for the corresponding sequence
            stream_data(x, stream_size, sequence_id, client)

        # iterate through inference service responses
        # that have been written to the client's out_q
        futures = []
        while len(timeseries) > 0:
            # see if the client has a response for us,
            # otherwise take a breath then keep going
            response = client.get()
            if response is None:
                continue
            y, _, sequence_id = response
            # update our running neural network outputs
            y = np.append(timeseries[sequence_id]["y"], y)
            timeseries[sequence_id]["y"] = y
            if len(timeseries[sequence_id]["y"]) == len(
                timeseries[sequence_id]["t"]
            ):
                logging.debug(f"Finished inference on sequence {sequence_id}")

                # grab the relevant data from the dictionary entry
                # and make sure that they have the same length
                ts = timeseries.pop(sequence_id)
                y, t = ts["y"], ts["t"]
                if len(y) != len(t):
                    raise ValueError(
                        "Sequence {} has completed but output array "
                        "has length {} which doesn't match length "
                        "of time array {}".format(sequence_id, len(y), len(t))
                    )

                # submit a write job to the process pool
                future = executor.submit(
                    write_timeseries, ts_write_dir, "out", t=t, y=y
                )
                futures.append(future)

        # wait for all the files from this timeslide to get written
        for fname in as_completed(futures):
            logging.debug(f"Wrote inferred segment to file '{fname}'")


@typeo
def main(
    model_repo_dir: str,
    model_name: str,
    data_dir: Path,
    write_dir: Path,
    fields: Iterable[str],
    sample_rate: float,
    inference_sampling_rate: float,
    num_workers: int,
    model_version: int = -1,
    base_sequence_id: int = 1001,
    log_file: Optional[Path] = None,
    verbose: bool = False,
):
    configure_logging(log_file, verbose)
    stream_size = int(sample_rate // inference_sampling_rate)

    if log_file is not None:
        server_log_file = log_file.parent / "server.log"
    else:
        server_log_file = None

    # spin up a triton server and don't move on until it's ready
    with serve(model_repo_dir, wait=True, log_file=server_log_file):
        # now build a client to connect to the inference service
        client = InferenceClient("localhost:8001", model_name, model_version)

        # create a process pool that we'll use to perform
        # read/writes of timeseries in parallel
        executor = AsyncExecutor(num_workers, thread=False)

        # now enter a context which will:
        # - for the client, start a streaming connection with
        #       with the inference service and launch a separate
        #       process for inference
        # - for the executor, launch the process pool
        with client, executor:
            for field in fields:
                # initialize all the `TimeSlide`s which will organize
                # their corresponding files into segments
                timeslides = [TimeSlide(i, field) for i in data_dir.iterdir()]

                # now actually do inference in a separate function since
                # we're already 2 contexts deep and we'll need to do
                # some nested looping on top of this
                infer(
                    client,
                    executor,
                    timeslides,
                    write_dir,
                    stream_size,
                    base_sequence_id,
                )


if __name__ == "__main__":
    main()
