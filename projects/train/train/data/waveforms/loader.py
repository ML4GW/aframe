from pathlib import Path

import h5py
import torch

from .sampler import WaveformSampler


class WaveformLoader(WaveformSampler):
    """
    Torch module for loading training and validation
    waveforms from disk and sampling them during training.
    TODO: modify this to sample waveforms from disk, taking
    an index sampler object so that DDP training can sample
    different waveforms for each device.
    Args:
        training_waveform_file:
            Path to the training waveforms file
    """

    def __init__(
        self,
        *args,
        training_waveform_file: Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.training_waveform_file = training_waveform_file

        with h5py.File(training_waveform_file) as f:
            self.num_train_waveforms = len(f["waveforms"]["cross"])

    def get_train_waveforms(self, world_size, rank, device):
        """
        Returns train waveforms for this device
        """
        start, stop = self.get_slice_bounds(
            self.num_train_waveforms, world_size, rank
        )
        with h5py.File(self.val_waveform_file) as f:
            waveforms = []
            for key in f["waveforms"].keys():
                waveforms.append(torch.Tensor(f["waveforms"][key][start:stop]))

            if (
                self.right_pad
                != f.attrs["duration"] - f.attrs["coalescence_time"]
            ):
                raise ValueError(
                    "Training and validation waveform files do not have "
                    "the same coalescence time and/or duration"
                )

        self.train_waveforms = torch.stack(waveforms, dim=0).to(device)

    def sample(self, X: torch.Tensor):
        """
        Sample method for generating training waveforms
        """
        N = len(X)
        idx = torch.randperm(self.num_train_waveforms)[:N]
        waveforms = self.train_waveforms[:, idx]

        hc, hp = waveforms
        return hc, hp
