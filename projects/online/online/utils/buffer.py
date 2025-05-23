import h5py
import numpy as np
import torch


class InputBuffer(torch.nn.Module):
    """
    A buffer for storing raw strain data for use
    in AMPLFI parameter estimation followup of events detected by Aframe

    Args:
        num_channels:
            The number of channels in the data
        sample_rate:
            The sampling rate of the data
        buffer_length:
            The length of the buffer in seconds
        amplfi_kernel_length:
            The length of the window to use for parameter estimation in seconds
        event_position:
            The placement of the coalescence time
            of the event in the AMPLFI analysis window in seconds
    """

    def __init__(
        self,
        ifos: list[str],
        sample_rate: float,
        buffer_length: float,
        fduration: float,
        amplfi_kernel_length: float,
        event_position: float,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.fduration = fduration
        self.num_channels = len(ifos)
        self.ifos = ifos
        self.sample_rate = sample_rate
        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * sample_rate)
        self.amplfi_kernel_length = amplfi_kernel_length
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
            for i, ifo in enumerate(self.ifos):
                f.create_dataset(ifo, data=self.input_buffer[i, :].cpu())

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

    def get_amplfi_data(
        self, event_time: float, ifos: list[str], psd_length: float
    ):
        window_start = (
            event_time - self.t0 - self.event_position - self.fduration / 2
        )
        window_start = int(self.sample_rate * window_start)
        window_end = int(
            window_start
            + (self.amplfi_kernel_length + self.fduration) * self.sample_rate
        )

        psd_start = window_start - int(psd_length * self.sample_rate)

        # get indices in tensor corresponding to requested ifos
        indices = torch.tensor([self.ifos.index(ifo) for ifo in ifos])

        psd_data = self.input_buffer[indices, psd_start:window_start]
        window = self.input_buffer[indices, window_start:window_end]

        return psd_data, window


class OutputBuffer(torch.nn.Module):
    """
    A buffer for storing raw and integrated neural network output

    Args:
        online_inference_rate:
            Rate at which Aframe's output is sampled online
        offline_inference_rate:
            Rate at which inference was performed offline when
            establishing the background and foreground distributions
        integration_window_length:
            The length of the integration window in seconds
        buffer_length:
            The length of the buffer in seconds
    """

    def __init__(
        self,
        online_inference_rate: float,
        offline_inference_rate: float,
        integration_window_length: float,
        buffer_length: float,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.online_inference_rate = online_inference_rate
        self.timing_integrator_size = (
            int(integration_window_length * online_inference_rate) + 1
        )
        self.timing_window = torch.ones(
            (1, 1, self.timing_integrator_size), device=device
        )
        self.timing_window /= self.timing_integrator_size

        significance_integrator_size = (
            int(integration_window_length * offline_inference_rate) + 1
        )
        self.significance_window = torch.ones(
            (1, 1, significance_integrator_size), device=device
        )
        self.significance_window /= significance_integrator_size

        self.online_offline_stride = int(
            online_inference_rate / offline_inference_rate
        )

        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * online_inference_rate)

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

    def write(self, path):
        start = self.t0
        stop = self.t0 + self.buffer_length - 1 / self.online_inference_rate
        time = np.linspace(start, stop, self.buffer_size)
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=time)
            f.create_dataset("output", data=self.output_buffer.cpu())
            f.create_dataset(
                "integrated",
                data=self.integrated_buffer.detach().cpu().numpy(),
            )

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        timing_output = torch.nn.functional.conv1d(
            x, self.timing_window, padding="valid"
        )
        x = x[..., :: self.online_offline_stride]
        significance_output = torch.nn.functional.conv1d(
            x, self.significance_window, padding="valid"
        )
        return timing_output[..., 1:], significance_output[..., 1:]

    def update(self, update: torch.Tensor, t0: float):
        # first append update to the output buffer
        # and remove buffer_size samples from front
        self.output_buffer = torch.cat([self.output_buffer, update])
        self.output_buffer = self.output_buffer[-self.buffer_size :]

        # t0 corresponds to the time of the first sample in the update
        # self.t0 corresponds to the earliest time in the buffer
        update_duration = len(update) / self.online_inference_rate
        self.t0 = t0 - (self.buffer_length - update_duration)

        integration_size = self.integrator_size + len(update)
        y = self.output_buffer[-integration_size:]
        timing_output, significance_output = self.integrate(y)
        self.integrated_buffer = torch.cat(
            [self.integrated_buffer, timing_output]
        )
        self.integrated_buffer = self.integrated_buffer[-self.buffer_size :]
        timing_cpu = timing_output.detach().cpu().numpy()
        significance_cpu = significance_output.detach().cpu().numpy()
        return timing_cpu, significance_cpu
