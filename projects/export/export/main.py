import logging
from typing import Optional

import s3fs
import torch
from export.snapshotter import add_streaming_input_preprocessor

import hermes.quiver as qv


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


def open_file(path, mode="rb"):
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        return fs.open(path, mode)
    else:
        # For local paths
        return open(path, mode)


def export(
    weights: str,
    repository_directory: str,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.TENSORRT,
    clean: bool = False,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Export a aframe architecture to a model repository
    for streaming inference, including adding a model
    for caching input snapshot state on the server.

    Args:
        weights:
            File Like object or Path representing
            a set of trained weights that will be
            exported to a model_repository. Supports
            local and S3 paths.
        repository_directory:
            Directory to which to save the models and their
            configs
        logdir:
            Directory to which logs will be written
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train aframe
        kernel_length:
            Length of segment in seconds that the network sees
        inference_sampling_rate:
            The rate at which kernels are sampled from the
            h(t) timeseries. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        batch_size:
            Number of kernels per batch
        fduration:
            Length of the time-domain whitening filter in seconds
        psd_length:
            Length of background time in seconds to use for PSD
            calculation
        fftlength:
            Length of time in seconds to use to calculate the FFT
            during whitening
        highpass:
            Frequency to use for a highpass filter
        streams_per_gpu:
            The number of snapshot states to host per GPU during
            inference
        aframe_instances:
            The number of concurrent execution instances of the
            aframe architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            aframe architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        **kwargs:
            Key word arguments specific to the export platform
    """

    # load in the model graph
    logging.info("Initializing model graph")
    with open_file(weights, "rb") as f:
        graph = nn = torch.jit.load(f)

    graph.eval()
    logging.info(f"Initialize:\n{nn}")

    # instantiate a model repository at the
    # indicated location. Split up the preprocessor
    # and the neural network (which we'll call aframe)
    # to export/scale them separately, and start by
    # seeing if either already exists in the model repo
    repo = qv.ModelRepository(repository_directory, clean)
    try:
        aframe = repo.models["aframe"]
    except KeyError:
        aframe = repo.add("aframe", platform=platform)

    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if aframe_instances is not None:
        scale_model(aframe, aframe_instances)

    size = int(kernel_length * sample_rate)
    input_shape = (batch_size, num_ifos, size)
    # the network will have some different keyword
    # arguments required for export depending on
    # the target inference platform
    # TODO: hardcoding these kwargs for now, but worth
    # thinking about a more robust way to handle this
    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13

        # turn off graph optimization because of this error
        # https://github.com/triton-inference-server/server/issues/3418
        aframe.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = False

    aframe.export_version(
        graph,
        input_shapes={"whitened": input_shape},
        output_names=["discriminator"],
        **kwargs,
    )

    # now try to create an ensemble that has a snapshotter
    # at the front for streaming new data to
    ensemble_name = "aframe-stream"
    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
        # if fftlength isn't specified, calculate the default value
        fftlength = fftlength or kernel_length + fduration
        whitened = add_streaming_input_preprocessor(
            ensemble,
            aframe.inputs["whitened"],
            psd_length=psd_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
            streams_per_gpu=streams_per_gpu,
        )
        ensemble.pipe(whitened, aframe.inputs["whitened"])

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        ensemble.add_output(aframe.outputs["discriminator"])
        ensemble.export_version(None)
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has aframe
        # and the snapshotter as a part of its models
        if aframe not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'aframe'".format(ensemble_name)
            )
        # TODO: checks for snapshotter and preprocessor

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()
