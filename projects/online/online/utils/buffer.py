import h5py
import numpy as np
import torch


class InputBuffer(torch.nn.Module):
    """
    A buffer for storing raw strain data f
    or use in parameter estimation (amplfi) followup
    of events detected by Aframe

    Args:
        num_channels:
            The number of channels in the data
        sample_rate:
            The sampling rate of the data
        buffer_length:
            The length of the buffer in seconds
        pe_window:
            The length of the window to use for parameter estimation in seconds
        event_position:
            The placement of the coalescence time
            of the event in the pe window in seconds
    """

    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        buffer_length: float,
        fduration: float,
        pe_window: float,
        event_position: float,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.fduration = fduration
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * sample_rate)
        self.pe_window = pe_window
        self.event_position = event_position

        self.input_buffer = torch.zeros(
            (self.num_channels, self.buffer_size), device=device
        )
        self.reset()

    def write(self, write_path, event_time):
        start = self.t0
        stop = self.t0 + self.buffer_length
        time = np.linspace(start, stop, self.buffer_size)
        with h5py.File(write_path, "w") as f:
            f.attrs.create("event_time", data=event_time)
            f.create_dataset("time", data=time)
            f.create_dataset("H1", data=self.input_buffer[0, :].cpu())
            f.create_dataset("L1", data=self.input_buffer[1, :].cpu())

    def reset(self):
        self.t0 = None
        self.input_buffer = torch.zeros(
            (self.num_channels, self.buffer_size), device=self.device
        )

    def update(self, update, t0):
        self.input_buffer = torch.cat([self.input_buffer, update], axis=-1)
        self.input_buffer = self.input_buffer[:, -self.buffer_size :]
        update_duration = update.shape[-1] / self.sample_rate
        self.t0 = t0 - (self.buffer_length - update_duration)

    def get_pe_data(self, event_time: float):
        window_start = (
            event_time - self.t0 - self.event_position - self.fduration / 2
        )
        window_start = int(self.sample_rate * window_start)
        window_end = int(
            window_start + (self.pe_window + self.fduration) * self.sample_rate
        )

        psd = self.input_buffer[:, :window_start]
        window = self.input_buffer[:, window_start:window_end]

        return psd, window


class OutputBuffer(torch.nn.Module):
    """
    A buffer for storing raw and integrated neural network output

    Args:
        inference_sampling_rate: The sampling rate of the neural network
        integration_window_length: The length of the integration window
        buffer_length: The length of the buffer in seconds
    """

    def __init__(
        self,
        inference_sampling_rate: float,
        integration_window_length: float,
        buffer_length: float,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.inference_sampling_rate = inference_sampling_rate
        self.integrator_size = int(
            integration_window_length * inference_sampling_rate
        )
        self.window = torch.ones((1, 1, self.integrator_size), device=device)
        self.window /= self.integrator_size
        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * inference_sampling_rate)

        self.output_buffer = torch.zeros((self.buffer_size,), device=device)

        self.reset()

    def reset(self):
        self.t0 = None
        self.output_buffer = torch.zeros(
            (self.buffer_size,), device=self.device
        )
        self.integrated_buffer = torch.zeros(
            (self.buffer_size,), device=self.device, requires_grad=False
        )

    def write(self, write_path, event_time):
        start = self.t0
        stop = self.t0 + self.buffer_length
        time = np.linspace(start, stop, self.buffer_size)
        with h5py.File(write_path, "w") as f:
            f.attrs.create("event_time", data=event_time)
            f.create_dataset("time", data=time)
            f.create_dataset("output", data=self.output_buffer.cpu())
            f.create_dataset(
                "integrated",
                data=self.integrated_buffer.detach().cpu().numpy(),
            )

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(x, self.window, padding="valid")
        return y[0, 0]

    def update(self, update: torch.Tensor, t0: float):
        # first append update to the output buffer
        # and remove buffer_size samples from front
        self.output_buffer = torch.cat([self.output_buffer, update])
        self.output_buffer = self.output_buffer[-self.buffer_size :]

        # t0 corresponds to the time of the first sample in the update
        # self.t0 corresponds to the earliest time in the buffer
        update_duration = len(update) / self.inference_sampling_rate
        self.t0 = t0 - (self.buffer_length - update_duration)

        integration_size = self.integrator_size + len(update)
        y = self.output_buffer[-integration_size:]
        integrated = self.integrate(y)
        self.integrated_buffer = torch.cat(
            [self.integrated_buffer, integrated]
        )
        self.integrated_buffer = self.integrated_buffer[-self.buffer_size :]
        return integrated.detach().cpu().numpy()
