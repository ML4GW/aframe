import torch

from utils.preprocessing import BackgroundSnapshotter


class OnlineSnapshotter(BackgroundSnapshotter):
    """
    Light subclass of BackgroundSnapshotter that
    registers the initial state as a buffer, and
    keeps track of contiguous update size to determine
    if there is enough data to calculate a PSD
    """

    def __init__(self, *args, update_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_size = update_size
        self.contiguous_update_size = 0
        self.register_buffer(
            "initial_state", torch.zeros((1, self.state_size))
        )

    @property
    def full_psd_present(self):
        return (
            self.contiguous_update_size >= self.state_size - self.update_size
        )

    def reset(self):
        self.contiguous_update_size = 0

    def forward(self, update, state):
        X, state = super().forward(update, state)
        if not self.full_psd_present:
            self.contiguous_update_size += self.update.shape[-1]
        return X, state
