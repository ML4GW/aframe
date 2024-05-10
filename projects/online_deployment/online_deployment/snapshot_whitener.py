from typing import Optional

import torch

from aframe.architectures import BackgroundSnapshotter, BatchWhitener


class SnapshotWhitener(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        psd_length: float,
        kernel_length: float,
        fduration: float,
        sample_rate: float,
        inference_sampling_rate: float,
        fftlength: float,
        highpass: Optional[float] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to("cuda")

        # Updates come in 1 second chunks, so each
        # update will generate a batch of
        # `inference_sampling_rate` overlapping
        # windows to whiten
        batch_size = 1 * inference_sampling_rate
        self.batch_whitener = BatchWhitener(
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
        ).to("cuda")

        self.step_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        self.filter_size = int(fduration * sample_rate)

        self.sample_rate = sample_rate
        self.contiguous_update_size = 0

    @property
    def state_size(self):
        return (
            self.psd_size
            + self.kernel_size
            + self.filter_size
            - self.step_size
        )

    def get_initial_state(self):
        self.contiguous_update_size = 0
        return torch.zeros((1, self.num_channels, self.state_size))

    def forward(self, update, current_state):
        update = update[None, :, :]
        X, current_state = self.snapshotter(update, current_state)
        # If we haven't had enough updates in a row to
        # meaningfully whiten, note that for upstream processes
        full_psd_present = (
            self.contiguous_update_size >= self.state_size - update.shape[-1]
        )
        if not full_psd_present:
            self.contiguous_update_size += update.shape[-1]
        return self.batch_whitener(X), current_state, full_psd_present
