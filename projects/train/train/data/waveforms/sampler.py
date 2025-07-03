from pathlib import Path

import h5py
import torch
from utils import x_per_y

Distribution = torch.distributions.Distribution


class WaveformSampler(torch.nn.Module):
    """
    Base object defining methods that waveform producing classes
    should implement. Should not be instantiated on its own.
    Args:
        fduration:
            Desired length in seconds of the time domain
            response of the whitening filter built from PSDs.
            See `ml4gw.spectral.truncate_inverse_power_spectrum`
        kernel_length:
            Length in seconds of window passed to neural network.
        sample_rate:
            Sample rate in Hz of generated waveforms
        val_waveform_file:
            Path to the validation waveforms file.
    """

    def __init__(
        self,
        *args,
        fduration: float,
        kernel_length: float,
        sample_rate: float,
        val_waveform_file: Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fduration = fduration
        self.kernel_length = kernel_length
        self.sample_rate = sample_rate
        self.val_waveform_file = val_waveform_file

        with h5py.File(val_waveform_file) as f:
            key = list(f["waveforms"].keys())[0]
            self.num_val_waveforms = len(f["waveforms"][key])
            self.right_pad = f.attrs["duration"] - f.attrs["coalescence_time"]

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
        with h5py.File(self.val_waveform_file) as f:
            waveforms = []
            for key in f["waveforms"].keys():
                waveforms.append(torch.Tensor(f["waveforms"][key][start:stop]))

        return torch.stack(waveforms, dim=0)

    def get_test_waveforms(self):
        raise NotImplementedError

    def sample(self):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
