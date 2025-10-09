from typing import TYPE_CHECKING, Optional

from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils

from utils.mm_preprocessing import BackgroundSnapshotter, mm_BatchWhitener
from collections.abc import Sequence

if TYPE_CHECKING:
    from hermes.quiver.model import EnsembleModel, ExposedTensor


def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


def mm_add_streaming_input_preprocessor(
    input_shapes: list,
    ensemble: "EnsembleModel",
    input: list,
    psd_length: float,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    fduration: float,
    fftlength: float,
    batch_size: int,
    resample_rates: Sequence[float], 
    kernel_lengths: Sequence[float], 
    high_passes: Sequence[float], 
    low_passes: Sequence[float],
    inference_sampling_rates: Sequence[float],
    starting_offsets: Sequence[int],
    num_ifos: int,
    q: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    preproc_instances: Optional[int] = None,
    streams_per_gpu: int = 1,
) -> "ExposedTensor":
    """Create a snapshotter model and add it to the repository"""

    augmentor = None

    snapshotter = BackgroundSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    )

    stride = int(sample_rate / inference_sampling_rate)
    state_shape = (2, num_ifos, snapshotter.state_size)
    input_shape = (2, num_ifos, batch_size * stride)
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
    preprocessor = mm_BatchWhitener(
        resample_rates = resample_rates, 
        kernel_lengths = kernel_lengths, 
        high_passes = high_passes, 
        low_passes = low_passes,
        inference_sampling_rates = inference_sampling_rates,
        starting_offsets = starting_offsets,
        num_ifos = num_ifos,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=batch_size,
        fduration=fduration,
        fftlength=fftlength,
    )
    preproc_model = ensemble.repository.add(
        "preprocessor", platform=Platform.TORCHSCRIPT
    )
    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if preproc_instances is not None:
        scale_model(preproc_model, preproc_instances)

    input_shape = streaming_model.outputs["strain"].shape
    preproc_model.export_version(
        preprocessor,
        input_shapes={"strain": input_shape},
        output_names=[f"whitened_{i}" for i in range(len(input_shapes))],
    )
    ensemble.pipe(
        streaming_model.outputs["strain"],
        preproc_model.inputs["strain"],
    )
    return [preproc_model.outputs[f"whitened_{i}"] for i in range(len(input_shapes))]