from typing import TYPE_CHECKING, Optional

from utils.preprocessing import (
    BackgroundSnapshotter,
    BatchWhitener,
)
from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils

if TYPE_CHECKING:
    from hermes.quiver.model import EnsembleModel, ExposedTensor


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