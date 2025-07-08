from pathlib import Path

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

        waveform_set = self.waveform_set_cls.read(training_waveform_file)
        if waveform_set.right_pad != self.right_pad:
            raise ValueError(
                "Training waveform file does not have the same "
                "right pad as validation waveform file"
            )
        self.num_train_waveforms = len(waveform_set)

    def get_train_waveforms(self, world_size, rank, device):
        """
        Returns train waveforms for this device
        """
        start, stop = self.get_slice_bounds(
            self.num_train_waveforms, world_size, rank
        )
        waveform_set = self.waveform_set_cls.read(self.training_waveform_file)
        waveforms = torch.Tensor(waveform_set.waveforms[start:stop])
        self.train_waveforms = waveforms.to(device)

    def sample(self, X: torch.Tensor):
        """
        Sample method for generating training waveforms
        """
        N = len(X)
        idx = torch.randperm(self.num_train_waveforms)[:N]
        waveforms = self.train_waveforms[:, idx]

        hc, hp = waveforms
        return hc, hp
