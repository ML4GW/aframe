import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamLayout:
    """Batch layout of a stream of `size` samples.

    Args:
        size:
            Number of samples total in the stream.
        batch_size:
            Number of outputs produced per batch.
        stride:
            Number of input samples between consecutive outputs.
    """

    size: int
    batch_size: int
    stride: int

    @property
    def step_size(self) -> int:
        """Samples per batch update."""
        return self.batch_size * self.stride

    @property
    def num_batches(self) -> int:
        """Total number of batches in the stream, rounded up to include
        a zero-padded final batch.
        """
        return math.ceil(self.size / self.step_size)

    @property
    def remainder(self) -> int:
        """Samples in the final batch if `step_size` doesn't evenly
        divide `size`, else 0."""
        return self.size % self.step_size

    @property
    def num_pad(self) -> int:
        """Padding samples appended to fill the final batch."""
        return self.num_batches * self.step_size - self.size

    @property
    def num_outputs(self) -> int:
        """Number of outputs, excluding those computed from the padding
        of the final batch."""
        num_pad_outputs = self.num_pad // self.stride
        return self.num_batches * self.batch_size - num_pad_outputs

    def batch_bounds(self, batch_idx: int):
        """Sample range for one batch, and whether it is the last one."""
        last = batch_idx == self.num_batches - 1
        start = batch_idx * self.step_size
        end = start + self.step_size
        if last and self.remainder:
            end = start + self.remainder
        return start, end, last


@dataclass(frozen=True)
class StreamOutputs:
    """How network outputs are integrated and timestamped.

    Args:
        inference_sampling_rate:
            Network outputs per second.
        fduration:
            Length of the whitening filter, in seconds.
        integration_window_length:
            Length of the tophat the outputs are averaged over, in seconds.
        psd_length:
            Seconds of data used to estimate the PSD for whitening.
    """

    inference_sampling_rate: float
    fduration: float
    integration_window_length: float
    psd_length: float

    @property
    def burn_in_size(self) -> int:
        """Outputs from the start of a stream before there was `psd_length`
        seconds of data."""
        return int(self.psd_length * self.inference_sampling_rate)

    @property
    def integration_size(self) -> int:
        """Number of outputs averaged by the tophat integration."""
        length = self.integration_window_length * self.inference_sampling_rate
        return int(length) + 1

    @property
    def lag(self) -> float:
        """Seconds between when an integrated output becomes available and
        its timestamp"""
        return self.fduration / 2 + self.integration_window_length

    def integrate(self, y: np.ndarray, first_output: int = 0) -> np.ndarray:
        """Integrate outputs with a tophat window.

        Args:
            y: Consecutive model outputs from a stream
            first_output: Index of `y[0]` in the full stream

        Outputs from the burn-in period are dropped. If `y[0]` is not
        the first index of the output stream, then the first
        `integration_size - 1` values are smaller than they should be.
        """
        burn_in = max(self.burn_in_size - first_output, 0)
        window = np.ones(self.integration_size) / self.integration_size
        integrated = np.zeros(len(y))
        y = y[burn_in:]
        integrated[burn_in:] = np.convolve(y, window)[: len(y)]
        return integrated

    def availability_time(self, t0: float, index: int | np.ndarray):
        """Time that that output at `index` of a stream starting at `t0`
        can be computed"""
        return t0 + (index + 1) / self.inference_sampling_rate

    def timestamp(self, t0: float, index: int | np.ndarray):
        """Timestamp of the integrated output `index` of a stream starting
        at `t0`"""
        return self.availability_time(t0, index) - self.lag
