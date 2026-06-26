from dataclasses import fields
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
from utils import x_per_y

from ledger.injections import WaveformSet, waveform_class_factory

Distribution = torch.distributions.Distribution


# TODO: Make this class ABC?
class WaveformSampler(torch.nn.Module):
    """
    Base object defining methods that waveform producing classes
    should implement. Should not be instantiated on its own.
    Args:
        ifos:
            List of interferometers that are being trained on.
        sample_rate:
            Sample rate in Hz of generated waveforms
        val_waveform_file:
            Path to the validation waveforms file.
    """

    def __init__(
        self,
        *args,
        ifos: List[str],
        sample_rate: float,
        val_waveform_file: Path,
        num_val_waveforms: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ifos = ifos
        self.sample_rate = sample_rate
        self.val_waveform_file = val_waveform_file

        # Read only metadata; reading the full ledger would load the entire
        # (potentially many-GB) validation file into memory. num_val_waveforms
        # optionally caps how many validation waveforms are used.
        with h5py.File(val_waveform_file, "r") as f:
            total = int(f.attrs["num_injections"])
            self.right_pad = float(f.attrs["right_pad"])
        self.num_val_waveforms = (
            total
            if num_val_waveforms is None
            else min(int(num_val_waveforms), total)
        )

    @property
    def waveform_set_cls(self):
        cls = waveform_class_factory(
            self.ifos,
            WaveformSet,
            "WaveformSet",
        )
        return cls

    def get_slice_bounds(self, total, world_size, rank) -> tuple[int, int]:
        """
        Determine waveform indices to load for this device
        given our rank and world size
        """
        per_dev = x_per_y(abs(total), world_size)
        start = rank * per_dev
        stop = (rank + 1) * per_dev
        return start, stop

    # Assuming that we're going to be loading validation waveforms
    # from disk for now, so this function can be defined here.
    def get_val_waveforms(
        self, world_size, rank
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns validation waveforms and injection parameters
        for this device.

        Returns:
            Tuple of (waveforms, params) where waveforms has shape
            ``(N, num_ifos, T)`` and params is a dict mapping each
            scalar injection parameter name to a tensor of
            shape ``(N, ...)``.
        """
        start, stop = self.get_slice_bounds(
            self.num_val_waveforms, world_size, rank
        )
        # Load only the [start:stop] slice rather than the whole file.
        idx = np.arange(start, stop)
        with h5py.File(self.val_waveform_file, "r") as h5f:
            waveform_set = self.waveform_set_cls._load_with_idx(h5f, idx)
        waveforms = torch.Tensor(waveform_set.waveforms)

        params: dict[str, torch.Tensor] = {}
        for f in fields(waveform_set):
            if f.metadata.get("kind") != "parameter":
                continue
            val = getattr(waveform_set, f.name)
            if isinstance(val, np.ndarray):
                params[f.name] = torch.from_numpy(val)

        return waveforms, params

    def get_test_waveforms(self):
        raise NotImplementedError

    def sample(self):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
