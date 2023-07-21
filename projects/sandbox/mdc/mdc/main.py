import logging
import time
from pathlib import Path
from typing import List, Optional

import h5py
from mdc.callback import Callback
from mdc.data import ChunkedSegmentLoader, batch_chunks
from typeo import scriptify

from aframe.logging import configure_logging
from hermes.aeriel.client import InferenceClient


@scriptify
def main(
    ip: str,
    model_name: str,
    data_dir: Path,
    start: int,
    output_dir: Path,
    sample_rate: float,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    fduration: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Perform inference using Triton on a directory
    of timeseries files, using a particular set of
    interferometer time shifts. Network outputs will
    be saved both as-is and using local integration.

    Args:
        ip:
            The IP address at which a Triton server
            hosting the indicated model is running
        model_name:
            The name of the model to which to make
            inference requests
        data_dir:
            Directory containing input files representing
            timeseries on which to perform inference.
            Each HDF5 file in this directory will be used
            for inference.
        write_dir:
            Directory to which to save raw and locally
            integrated network outputs.
        sample_rate:
            Rate at which input timeseries data has
            been sampled.
        inference_sampling_rate:
            The rate at which to sample windows for
            inference from the input timeseries.
            Corresponds to the sample rate of the
            output timeseries.
        batch_size:
            The number of subsequent windows to
            include in a single batch of inference.
        window_length:
            Length of the window over which network
            outputs should be locally integrated,
            specified in seconds.
        max_shift:
            The maximum shift value across all runs
            in the timeslide analysis that this run
            is a part of. This helps keep all output
            timeseries the same length.
        throughput:
            Rate at which to make requests, in units
            of seconds of data per second `[s' / s]`.
        sequence_id:
            Identifier to assign to all the sequences
            of inference requests in this run to match up
            with a corresponding snapshot state on the
            inference server.
        model_version:
            Version of the model from which to request
            inference. Default value of `-1` indicates
            the latest available version of the model.
        log_file:
            File to which to write inference logs.
        verbose:
            Flag controlling whether logging verbosity
            is `DEBUG` (`True`) or `INFO` (`False`)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    configure_logging(log_file, verbose)

    callback = Callback(
        id=sequence_id,
        sample_rate=inference_sampling_rate,
        batch_size=batch_size,
        integration_window_length=integration_window_length,
        cluster_window_length=cluster_window_length,
        fduration=fduration,
    )

    # TODO: what's the best place to infer this info?
    # Obviously in the data loader, but how to communicate
    # to the batcher/callback?
    with h5py.File(data_dir / "background.hdf", "r") as f:
        size = len(f[ifos[0]][str(start)])
    duration = size / sample_rate
    end = start + duration

    logging.info(f"Connecting to server at {ip}:8001")
    client = InferenceClient(
        f"{ip}:8001", model_name, model_version, callback=callback
    )
    chunk_size = int(chunk_size * sample_rate)
    loader = ChunkedSegmentLoader(data_dir, str(start), ifos, chunk_size)
    with client, loader as loader:
        logging.info(
            f"Beginning inference on {duration}s sequence {start}-{end}"
        )
        num_steps = callback.initialize(start, end)

        # create an iterator that will break
        # these chunks up into update-sized pieces
        batcher = batch_chunks(
            loader,
            num_steps,
            batch_size,
            inference_sampling_rate,
            sample_rate,
            throughput,
        )

        for i, (background, injected) in enumerate(batcher):
            client.infer(
                background,
                request_id=i,
                sequence_id=sequence_id,
                sequence_start=i == 0,
                sequence_end=i == (num_steps - 1),
            )
            client.infer(
                injected,
                request_id=i,
                sequence_id=sequence_id + 1,
                sequence_start=i == 0,
                sequence_end=i == (num_steps - 1),
            )

            # wait for the first response to come back
            # before proceeding in case the snapshot
            # state requires some warm up
            if not i:
                logging.debug("Waiting for initial response")
                callback.wait()

        result = client.get()
        while result is None:
            time.sleep(1e-1)
            result = client.get()
        background, foreground = result

    logging.info("Completed inference")
    logging.info("Building event sets and writing to files")
    background.write(output_dir / "background.hdf")
    foreground.write(output_dir / "foreground.hdf")


if __name__ == "__main__":
    main()
