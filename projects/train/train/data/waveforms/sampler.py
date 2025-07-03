from pathlib import Path
from typing import List

import torch
from utils import x_per_y

from ledger.injections import WaveformSet, waveform_class_factory

Distribution = torch.distributions.Distribution


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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ifos = ifos
        self.sample_rate = sample_rate
        self.val_waveform_file = val_waveform_file

        waveform_set = self.waveform_set_cls.read(val_waveform_file)
        self.num_val_waveforms = len(waveform_set)
        self.right_pad = waveform_set.right_pad

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
    def get_val_waveforms(self, world_size, rank):
        """
        Returns validation waveforms for this device
        """
        start, stop = self.get_slice_bounds(
            self.num_val_waveforms, world_size, rank
        )
        waveform_set = self.waveform_set_cls.read(self.val_waveform_file)
        return torch.Tensor(waveform_set.waveforms[start:stop])

    def get_test_waveforms(self):
        raise NotImplementedError

    def sample(self):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
