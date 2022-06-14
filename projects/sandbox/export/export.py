import logging
from pathlib import Path
from typing import Callable, Optional

import hermes.quiver as qv
import torch

from bbhnet.architectures import architecturize
from bbhnet.data.transforms import WhiteningTransform
from bbhnet.logging import configure_logging


@architecturize
def export(
    architecture: Callable,
    repository_directory: str,
    output_directory: Path,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    weights: Optional[Path] = None,
    streams_per_gpu: int = 1,
    instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    clean: bool = False,
    verbose: bool = False,
) -> None:
    """
    Export a BBHNet architecture to a model repository
    for streaming inference, including adding a model
    for caching input snapshot state on the server.
    Args:
        architecture:
            A function which takes as input a number of witness
            channels and returns an instantiated torch `Module`
            which represents a DeepClean network architecture
        repository_directory:
            Directory to which to save the models and their
            configs
        output_directory:
            Path to save logs. If `weights` is `None`, this
            directory is assumed to contain a file `"weights.pt"`.
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train BBHNet
        kernel_length:
            The length, in seconds, of the input to DeepClean
        inference_sampling_rate:
            The rate at which kernels are sampled from the
            h(t) timeseries. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        weights:
            Path to a set of trained weights with which to
            initialize the network architecture. If left as
            `None`, a file called `"weights.pt"` will be looked
            for in the `output_directory`.
        streams_per_gpu:
            The number of snapshot states to host per GPU during
            inference
        instances:
            The number of concurrent execution instances of the
            BBHNet architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            DeepClean architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If set, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
    """

    # if we didn't specify a weights filename, assume
    # that a "weights.pt" lives in our output directory
    if weights is None or weights.is_dir():
        weights_dir = output_directory if weights is None else weights
        weights = weights_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights file '{weights}'")

    configure_logging(output_directory / "export.log", verbose)

    # instantiate the architecture and initialize
    # its weights with the trained values
    logging.info(f"Creating model and loading weights from {weights}")
    nn = architecture(num_ifos)

    # hardcoding preprocessing, what's the best way to handle?
    preprocessor = WhiteningTransform(num_ifos, sample_rate, kernel_length)
    nn = torch.nn.Sequential(preprocessor, nn)
    nn.load_state_dict(torch.load(weights))
    nn.eval()

    # instantiate a model repository at the
    # indicated location and see if a deepclean
    # model already exists in this repository
    repo = qv.ModelRepository(repository_directory)
    if clean:
        logging.info(f"Cleaning model repository {repository_directory}")
        for model in repo.models:
            logging.info(f"Removing model {model}")
            repo.remove(model)

    try:
        bbhnet = repo.models["bbhnet"]

        # if the model exists already and we specified
        # a concurrent instance count, scale the existing
        # instance group to this value
        if instances is not None:
            # TODO: should quiver handle this under the hood?
            try:
                bbhnet.config.scale_instance_group(instances)
            except ValueError:
                bbhnet.config.add_instance_group(count=instances)
    except KeyError:
        # otherwise create the model using the indicated
        # platform and set up an instance group with the
        # indicated scale if one was provided
        bbhnet = repo.add("bbhnet", platform=platform)
        if instances is not None:
            bbhnet.config.add_instance_group(count=instances)

    # export this version of the model (with its current
    # weights), to this entry in the model repository
    input_shape = (1, num_ifos, int(kernel_length * sample_rate))
    bbhnet.export_version(
        nn, input_shapes={"hoft": input_shape}, output_names=["discriminator"]
    )

    # now try to create an ensemble that has a snapshotter
    # at the front for streaming new data to
    ensemble_name = "bbhnet-stream"
    stream_size = int(sample_rate / inference_sampling_rate)

    # see if we have an existing snapshot model up front,
    # since we'll make different choices later depending
    # on whether it already exists or not
    try:
        snapshotter = repo.models["snapshotter"]
    except KeyError:
        snapshotter = None

    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)

        # if a snapshot model already exists, add it to
        # the ensemble and pipe its output to the input of
        # bbhnet, otherwise use the `add_streaming_inputs`
        # method on the ensemble to create a snapshotter
        # and perform this piping for us
        if snapshotter is not None:
            ensemble.add_input(snapshotter.inputs["stream"])
            ensemble.pipe(
                snapshotter.outputs["snapshotter"], bbhnet.inputs["hoft"]
            )
        else:
            # there's no snapshotter, so make one
            ensemble.add_streaming_inputs(
                inputs=[bbhnet.inputs["hoft"]],
                stream_size=stream_size,
                name="snapshotter",
                streams_per_gpu=streams_per_gpu,
            )
            snapshotter = repo.models["snapshotter"]

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        ensemble.add_output(bbhnet.outputs["discriminator"])
        ensemble.export_version(None)
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has BBHNet
        # and the snapshotter as a part of its models
        if bbhnet not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'bbhnet'".format(ensemble_name)
            )
        elif snapshotter is None or snapshotter not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'snapshotter'".format(ensemble_name)
            )

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e9
    )
    snapshotter.config.write()


if __name__ == "__main__":
    export()
