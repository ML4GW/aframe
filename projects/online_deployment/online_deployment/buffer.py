import h5py
import numpy as np
import torch


class DataBuffer:
    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        inference_sampling_rate: float,
        integration_window_length: float,
        input_buffer_length: float,
        output_buffer_length: float,
    ):
        self.input_buffer = InputBuffer(
            num_channels=num_channels,
            sample_rate=sample_rate,
            buffer_length=input_buffer_length,
        )
        self.output_buffer = OutputBuffer(
            inference_sampling_rate=inference_sampling_rate,
            integration_window_length=integration_window_length,
            buffer_length=output_buffer_length,
        )

    def reset_state(self):
        self.input_buffer.reset_state()
        self.output_buffer.reset_state()

    def write(self, write_path, event_time):
        event_dir = f"event_{int(event_time)}"
        input_fname = write_path / event_dir / "strain.h5"
        output_fname = write_path / event_dir / "network_output.h5"

        self.input_buffer.write(input_fname, event_time)
        self.output_buffer.write(output_fname, event_time)

    def update(
        self,
        input_update,
        output_update,
        t0,
        input_time_offset,
        output_time_offset,
    ):
        self.input_buffer.update(input_update, t0 + input_time_offset)
        return self.output_buffer.update(
            output_update, t0 + output_time_offset
        )


class InputBuffer:
    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        buffer_length: float,
    ):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * sample_rate)
        self.reset_state()

    def write(self, write_path, event_time):
        start = self.t0
        stop = self.t0 + self.buffer_length
        time = np.linspace(start, stop, self.buffer_size)
        with h5py.File(write_path, "w") as f:
            f.attrs.create("event_time", data=event_time)
            f.create_dataset("time", data=time)
            f.create_dataset("H1", data=self.input_buffer[0, :].cpu())
            f.create_dataset("L1", data=self.input_buffer[1, :].cpu())

    def reset_state(self):
        self.t0 = 0
        self.input_buffer = torch.zeros(
            (self.num_channels, self.buffer_size), device="cuda"
        )

    def update(self, update, t0):
        self.input_buffer = torch.cat([self.input_buffer, update], axis=-1)
        self.input_buffer = self.input_buffer[:, -self.buffer_size :]
        update_duration = update.shape[-1] / self.sample_rate
        self.t0 = t0 - (self.buffer_length - update_duration)


class OutputBuffer:
    def __init__(
        self,
        inference_sampling_rate: float,
        integration_window_length: float,
        buffer_length: float,
    ):
        self.inference_sampling_rate = inference_sampling_rate
        self.integrator_size = int(
            integration_window_length * inference_sampling_rate
        )
        self.window = torch.ones((1, 1, self.integrator_size), device="cuda")
        self.window /= self.integrator_size
        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * inference_sampling_rate)
        self.reset_state()

    def reset_state(self):
        self.t0 = 0
        self.output_buffer = torch.zeros((self.buffer_size,), device="cuda")
        self.integrated_buffer = torch.zeros(
            (self.buffer_size,), device="cuda"
        )

    def write(self, write_path, event_time):
        start = self.t0
        stop = self.t0 + self.buffer_length
        time = np.linspace(start, stop, self.buffer_size)
        with h5py.File(write_path, "w") as f:
            f.attrs.create("event_time", data=event_time)
            f.create_dataset("time", data=time)
            f.create_dataset("output", data=self.output_buffer.cpu())
            f.create_dataset("integrated", data=self.integrated_buffer.cpu())

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(x, self.window, padding="valid")
        return y[0, 0]

    def update(self, update, t0):
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
        return integrated.cpu().numpy()
