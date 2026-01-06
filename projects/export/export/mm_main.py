import io
import logging
from typing import Optional

import h5py
import hermes.quiver as qv
import torch
from collections.abc import Sequence

from export.mm_snapshotter import mm_add_streaming_input_preprocessor
from utils.s3 import open_file
import os

from export.mm_modules import concatenation_layer

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


def mm_export(
    weights: str,
    repository_directory: str,
    batch_file: str,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    resample_rates: tuple, 
    kernel_lengths: tuple, 
    high_passes: tuple, 
    low_passes: tuple,
    inference_sampling_rates: tuple,
    starting_offsets: tuple,
    classes: tuple,
    fftlength: Optional[float] = None,
    q: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    preproc_instances: Optional[int] = None,
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
        batch_file:
            Path to file containing a batch of data from model
            training. This is used to determine the input size
            of the model. File structure is assumed to match
            the structure of the file written during training
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
        lowpass:
            Frequency to use for a lowpass filter
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
    resample_rates = list(resample_rates)
    kernel_lengths = list(kernel_lengths)
    high_passes = list(high_passes)
    low_passes = list(low_passes)
    inference_sampling_rates = list(inference_sampling_rates)
    starting_offsets = list(starting_offsets)
    classes = list(classes)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # load in the model graph
    logging.info("Initializing model graph")
    
    with open_file(weights, "rb") as f:
        graph = torch.jit.load(f, map_location="cpu")
    
    graph.eval()
    logging.info(f"Initialize:\n{graph}")
    
    with open_file(batch_file, "rb") as f:
        batch_file = h5py.File(io.BytesIO(f.read()))
    
    layers = sorted(batch_file.keys() - "y")
    input_shapes = [(batch_size*inference_sampling_rates[i]//max(inference_sampling_rates), 
                     batch_file[layer].shape[-2], 
                     batch_file[layer].shape[-1]) for i, layer in enumerate(layers)]
    n_layers = len(layers)
    
    graphs = []
    model_parent_dir = os.path.dirname(weights)
    for i in range(n_layers):
        with open_file(os.path.join(model_parent_dir, f"resnets_{i}.pt"), "rb") as f:
            graphs.append(torch.jit.load(f, map_location="cpu"))
            graphs[-1].eval()
    
    with open_file(os.path.join(model_parent_dir, f"fc.pt"), "rb") as f:
        fc = torch.jit.load(f, map_location="cpu")
        fc.eval()
    # instantiate a model repository at the
    # indicated location. Split up the preprocessor
    # and the neural network (which we'll call aframe)
    # to export/scale them separately, and start by
    # seeing if either already exists in the model repo
    
    repo = qv.ModelRepository(repository_directory, clean)
    aframe = []
    for i in range(n_layers):
        try:
            aframe.append(repo.models[f"resnet_{i}"])
        except KeyError:
            aframe.append(repo.add(f"resnet_{i}", platform=platform))
    
    try:
        aframe.append(repo.models["fc"])
    except KeyError:
        aframe.append(repo.add("fc", platform=platform))
    
    try:
        concatenation = repo.models["concatenation_layer"]
    except KeyError:
        concatenation = repo.add("concatenation_layer", platform=platform)
    
    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    #if aframe_instances is not None:
    #    scale_model(aframe, aframe_instances)
    
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
    
    for i in range(n_layers):
        aframe[i].export_version(
            graphs[i],
            input_shapes={f"whitened_{i}": input_shapes[i]},
            output_names=[f"classes_{i}"],
            **kwargs,
        )
    cl  = concatenation_layer(inference_sampling_rates)
    concatenation.export_version(
        cl,
        input_shapes={f"classes_{i}": (input_shapes[i][0], classes[i]) 
                      for i in range(n_layers)},
        output_names=["concatenated"],
        **kwargs,
    )
    
    aframe[-1].export_version(
        fc,
        input_shapes={"concatenated": (batch_size, sum(classes))},
        output_names=["discriminator"],
        **kwargs,
    )
    
    # now try to create an ensemble that has a snapshotter
    # at the front for streaming new data to
    ensemble_name = "aframe-stream"
    # if we don't, create one
    ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
    # if fftlength isn't specified, calculate the default value
    fftlength = fftlength or kernel_length + fduration
    whitened = mm_add_streaming_input_preprocessor(
        input_shapes = input_shapes,
        ensemble = ensemble,
        input = [aframe[i].inputs[f"whitened_{i}"] for i in range(n_layers)],
        psd_length=psd_length,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        q=q,
        batch_size=batch_size,
        highpass=highpass,
        lowpass=lowpass,
        preproc_instances=preproc_instances,
        streams_per_gpu=streams_per_gpu,
        resample_rates = resample_rates, 
        kernel_lengths = kernel_lengths, 
        high_passes = high_passes, 
        low_passes = low_passes,
        inference_sampling_rates = inference_sampling_rates,
        starting_offsets = starting_offsets,
        num_ifos = num_ifos,
    )
    for i in range(n_layers):
        ensemble.pipe(whitened[i], aframe[i].inputs[f"whitened_{i}"])
    
    for i in range(n_layers):
        ensemble.pipe(aframe[i].outputs[f"classes_{i}"], concatenation.inputs[f"classes_{i}"])
    
    ensemble.pipe(concatenation.outputs["concatenated"], aframe[-1].inputs["concatenated"])
    ensemble.add_output(aframe[-1].outputs["discriminator"])
    # export the ensemble model, which basically amounts
    # to writing its config and creating an empty version entry
    ensemble.export_version(None)
    # if there does already exist an ensemble by
    # the given name, make sure it has aframe
    # and the snapshotter as a part of its models

    # TODO: checks for snapshotter and preprocessor

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()