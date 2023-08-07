from typing import TYPE_CHECKING, Optional, Tuple

import torch

from aframe.architectures.preprocessor import Whitener
from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils
from ml4gw.transforms import SpectralDensity
from ml4gw.utils.slicing import unfold_windows

if TYPE_CHECKING:
    from hermes.quiver.model import EnsembleModel, ExposedTensor


class BackgroundSnapshotter(torch.nn.Module):
    """Update a kernel with a new piece of streaming data"""

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(
        self, update: torch.Tensor, snapshot: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot


class BatchWhitener(torch.nn.Module):
    """Calculate the PSDs and whiten an entire batch of kernels at once"""

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)

        length = (batch_size - 1) / inference_sampling_rate
        length += kernel_length + fduration
        self.size = int(length * sample_rate)

        self.spectral_density = SpectralDensity(
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="mean",
            fast=highpass is not None,
        )
        self.whitener = Whitener(fduration, sample_rate, highpass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = [x.shape[-1] - self.size, self.size]
        background, x = torch.split(x, splits, dim=-1)
        psd = self.spectral_density(background.double())
        x = self.whitener(x.double(), psd)
        x = unfold_windows(x, self.kernel_size, self.stride_size)
        return x[:, 0]


def add_streaming_input_preprocessor(
    ensemble: "EnsembleModel",
    input: "ExposedTensor",
    psd_length: float,
    sample_rate: float,
    inference_sampling_rate: float,
    fduration: float,
    fftlength: float,
    highpass: Optional[float] = None,
    streams_per_gpu: int = 1,
) -> "ExposedTensor":
    """Create a snapshotter model and add it to the repository"""

    batch_size, num_ifos, kernel_size = input.shape
    snapshotter = BackgroundSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_size / sample_rate,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    )

    stride = int(sample_rate / inference_sampling_rate)
    state_shape = (1, num_ifos, snapshotter.state_size)
    input_shape = (1, num_ifos, batch_size * stride)
    streaming_model = streaming_utils.add_streaming_model(
        ensemble.repository,
        streaming_layer=snapshotter,
        name="snapshotter",
        input_name="stream",
        input_shape=input_shape,
        state_names=["snapshot"],
        state_shapes=[state_shape],
        output_names=["strain"],
        streams_per_gpu=streams_per_gpu,
    )
    ensemble.add_input(streaming_model.inputs["stream"])

    preprocessor = BatchWhitener(
        kernel_size / sample_rate,
        sample_rate,
        batch_size=batch_size,
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
    )
    preproc_model = ensemble.repository.add(
        "preprocessor", platform=Platform.TORCHSCRIPT
    )

    input_shape = streaming_model.outputs["strain"].shape
    preproc_model.export_version(
        preprocessor, input_shapes=[input_shape], output_names=None
    )
    ensemble.pipe(
        streaming_model.outputs["strain"],
        preproc_model.inputs["INPUT__0"],
    )
    return preproc_model.outputs["OUTPUT__0"]
