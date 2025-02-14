from typing import TYPE_CHECKING, Optional

from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils
from ml4gw.transforms import SingleQTransform

from utils.preprocessing import BackgroundSnapshotter, BatchWhitener

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


def add_streaming_input_preprocessor(
    ensemble: "EnsembleModel",
    input: "ExposedTensor",
    psd_length: float,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    fduration: float,
    fftlength: float,
    q: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    preproc_instances: Optional[int] = None,
    streams_per_gpu: int = 1,
) -> "ExposedTensor":
    """Create a snapshotter model and add it to the repository"""

    batch_size, num_ifos, *kernel_size = input.shape
    if q is not None:
        if len(kernel_size) != 2:
            raise ValueError(
                "If q is not None, the input kernel should be 2D, "
                f"got {len(kernel_size)} dimension(s)"
            )
        augmentor = SingleQTransform(
            duration=kernel_length,
            sample_rate=sample_rate,
            spectrogram_shape=kernel_size,
            q=q,
        )
    else:
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
    preprocessor = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=batch_size,
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
        lowpass=lowpass,
        augmentor=augmentor,
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
        output_names=["whitened"],
    )
    ensemble.pipe(
        streaming_model.outputs["strain"],
        preproc_model.inputs["strain"],
    )
    return preproc_model.outputs["whitened"]
