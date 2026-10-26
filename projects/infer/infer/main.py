import time

import numpy as np
from hermes.aeriel.client import InferenceClient
from ledger.events import EventSet
from tqdm import tqdm

from infer.data import RnPStreamSequence, ShiftBatchSequence
from infer.local import LocalInferenceClient
from infer.postprocess import Postprocessor


def _throttle(deadline: float, interval: float) -> float:
    """Sleep until deadline, then return the next one interval later.

    Time already spent this iteration counts toward the interval, so a slow
    iteration shortens or eliminates the sleep.
    """
    now = time.monotonic()
    if now < deadline:
        time.sleep(deadline - now)
    return max(now, deadline) + interval


def infer_local(
    client: LocalInferenceClient,
    sequence: ShiftBatchSequence | RnPStreamSequence,
    postprocessors: list[Postprocessor],
) -> list[tuple[EventSet, EventSet, np.ndarray, np.ndarray | None]]:
    """Run inference in-process on every stream of a sequence at once."""
    client.reset()
    outputs = []
    for update in tqdm(sequence):
        outputs.append(client.infer_streams(update, sequence.psd_source))

    return sequence.results(np.concatenate(outputs, axis=-1), postprocessors)


class TritonOutputs:
    """Collect a Triton server's responses for every stream of a sequence."""

    def __init__(
        self, num_streams: int, num_steps: int, batch_size: int, base_id: int
    ):
        """
        Args:
            num_streams: Number of streams in each update from the sequence
            num_steps: The total number of updates in a sequence
            batch_size: Number of outputs in each response
            base_id: Sequence ID of stream 0. Stream i uses base_id + i.
        """
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.base_id = base_id

        # Allocate one row of outputs per stream and keep track of which
        # streams have had a response and which have had their final one.
        self.outputs = np.zeros((num_streams, num_steps * batch_size))
        self._started = {base_id + i: False for i in range(num_streams)}
        self._done = {base_id + i: False for i in range(num_streams)}

    @property
    def started(self) -> bool:
        return all(self._started.values())

    @property
    def done(self) -> bool:
        return all(self._done.values())

    def __call__(
        self, y: np.ndarray, request_id: int, sequence_id: int
    ) -> np.ndarray | None:
        """
        Once all outputs are present, returns those outputs in an array
        with one row per stream.

        Args:
            y: Outputs of one request, of shape `(batch_size, 1)`
            request_id: Step at which the request was sent
            sequence_id: the base_id plus the index of the request's stream

        Returns:
            If all `(num_streams, num_steps * batch_size)` outputs are present,
            they are returned. Otherwise None.
        """
        # Write the response into its stream's row with this step's offset.
        stream = sequence_id - self.base_id
        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self.outputs[stream, start:stop] = y[:, 0]

        # Record that the stream has started, and whether this was the
        # response to its last step.
        self._started[sequence_id] = True
        if stop == self.outputs.shape[-1]:
            self._done[sequence_id] = True

        if self.done:
            return self.outputs
        return None


def infer_triton(
    client: InferenceClient,
    sequence: ShiftBatchSequence | RnPStreamSequence,
    postprocessors: list[Postprocessor],
    base_id: int,
    rate: float | None = None,
) -> list[tuple[EventSet, EventSet, np.ndarray, np.ndarray | None]]:
    """
    Run inference on every stream of a sequence through a Triton server.

    Args:
        client:
            A hermes InferenceClient, which has already been started with
            `with client:`.
        sequence:
            A stream sequence that yields updates of shape `(N, C, step)`.
        postprocessors:
            List of N postprocessors, one per stream.
        base_id:
            Non-zero sequence ID of stream 0. Stream i uses base_id + i.
        rate:
            Average number of requests per second, or None for no limit.

    Returns:
        A list of tuples, one tuple per shift. Each tuple holds
        `(background events, recovered injections, background outputs,
          foreground outputs or None)`.
    """
    psd_source = sequence.psd_source
    num_streams = len(psd_source)
    triton_outputs = TritonOutputs(
        num_streams,
        len(sequence),
        sequence.stream_layout.batch_size,
        base_id,
    )
    client.callback = triton_outputs

    # Each step sends one request per stream, so space the steps
    # num_streams / rate seconds apart to average `rate` requests per
    # second.
    interval = num_streams / rate if rate is not None else 0
    deadline = time.monotonic()

    # Pass each stream to the server in the expected format: a pair of arrays,
    # stacked, with the first being used for PSD estimation and the second
    # actually being handed to the model.
    for step, update in enumerate(tqdm(sequence)):
        sequence_start = step == 0
        sequence_end = step == len(sequence) - 1
        deadline = _throttle(deadline, interval)
        for i in range(num_streams):
            client.infer(
                np.stack([update[psd_source[i]], update[i]], axis=0),
                request_id=step,
                sequence_id=base_id + i,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )

        if sequence_start:
            while not triton_outputs.started:
                client.get()
                time.sleep(1e-2)

    # Wait for the callback to give back the full outputs
    outputs = client.get()
    while outputs is None:
        outputs = client.get()
        time.sleep(1e-1)

    return sequence.results(outputs, postprocessors)
