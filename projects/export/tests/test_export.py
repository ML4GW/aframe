import re
import shutil
from contextlib import nullcontext
from pathlib import Path

import h5py
import hermes.quiver as qv
import numpy as np
import pytest
import torch
from google.protobuf import text_format
from ml4gw.nn.resnet import ResNet1D
from tritonclient.grpc.model_config_pb2 import ModelConfig

from export.main import export


# set up a directory for the entirety of the session
# which will store all the weight values of each
# NN we need to create in response to a particular
# num_ifos/sample_rate/kernel_length combination
@pytest.fixture(scope="session")
def weights_dir():
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(exist_ok=True, parents=True)
    yield weights_dir
    shutil.rmtree(weights_dir)


@pytest.fixture
def batch_file(tmp_path, batch_size, num_ifos, sample_rate, kernel_length):
    data = np.random.randn(batch_size, num_ifos, sample_rate * kernel_length)
    file = tmp_path / "batch_file.h5"
    with h5py.File(file, "w") as f:
        f.create_dataset("X", data=data)
    return file


@pytest.fixture
def platform():
    return qv.Platform.ONNX


# only create a new neural network if the weights for
# a network of this num_ifos/sample_rate/kernel_length
# combination has not yet been created. Otherwise just
# return the path to those weights as-is
@pytest.fixture
def architecture():
    return lambda num_ifos: ResNet1D(num_ifos, layers=[2, 2], classes=1)


@pytest.fixture
def trace_network_weights(weights_dir, architecture):
    def fn(num_ifos, target, sample_rate, kernel_length, batch_size):
        sample_input = torch.randn(
            batch_size, num_ifos, sample_rate * kernel_length
        )
        weights = weights_dir / f"{num_ifos}-{sample_rate}-{kernel_length}.pt"
        if not weights.exists():
            aframe = architecture(num_ifos)
            trace = torch.jit.trace(aframe, sample_input)
            with open(weights, "wb") as f:
                torch.jit.save(trace, f)

        shutil.copy(weights, target)

    return fn


@pytest.fixture
def repo_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def output_dir(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def fduration():
    return 1


def load_config(config_path: Path):
    config = ModelConfig()
    text_format.Merge(config_path.read_text(), config)
    return config


@pytest.fixture
def validate_repo(repo_dir):
    def fn(
        expected_aframe_instances,
        expected_snapshots,
        expected_versions,
        expected_num_ifos,
        expected_stream_size,
        expected_state_size,
        expected_aframe_size,
        expected_crop,
        expected_batch_size,
    ):
        for i, model in enumerate(repo_dir.iterdir()):
            config = load_config(model / "config.pbtxt")
            if model.name == "snapshotter":
                try:
                    instance_group = config.instance_group[0]
                except IndexError:
                    if expected_snapshots is not None:
                        raise ValueError(
                            "No instance group but expected snapshots "
                            f"is {expected_snapshots}"
                        )
                else:
                    if expected_snapshots is None:
                        raise ValueError(
                            "Didn't expect snapshots but found "
                            f"instance group {instance_group}"
                        )
                    assert instance_group.count == expected_snapshots

                # we now pass two streams:
                # one is used to whiten the other
                assert config.input[0].dims == [
                    2,
                    expected_num_ifos,
                    expected_stream_size * expected_batch_size,
                ]
                assert config.output[0].dims == [
                    2,
                    expected_num_ifos,
                    expected_state_size
                    + (expected_stream_size * expected_batch_size),
                ]

                assert (model / "1" / "model.onnx").exists()
                assert not (model / "2").is_dir()
            elif model.name == "aframe":
                expected_instances = expected_aframe_instances
                expected_output_shape = [expected_batch_size, 1]
                assert config.optimization.graph.level == -1

                try:
                    instance_group = config.instance_group[0]
                except IndexError:
                    if expected_instances is not None:
                        raise ValueError(
                            "No instance group but expected instances "
                            f"is {expected_instances}"
                        )
                else:
                    if expected_instances is None:
                        raise ValueError(
                            "Didn't expect aframe instances but found "
                            f"instance group {instance_group}"
                        )
                    assert instance_group.count == expected_instances

                assert config.input[0].dims == [
                    expected_batch_size,
                    expected_num_ifos,
                    expected_aframe_size,
                ]
                assert config.output[0].dims == expected_output_shape

                if isinstance(expected_versions, tuple):
                    idx = 0
                    versions = expected_versions[idx]
                else:
                    versions = expected_versions

                for j in range(versions):
                    assert (model / str(j + 1) / "model.onnx").is_file()
                assert not (model / str(j + 2)).is_dir()

            elif model.name == "aframe-stream":
                assert (model / "1").is_dir()
                assert not (model / "2").is_dir()
            elif model.name == "preprocessor":
                assert config.input[0].dims == [
                    2,
                    expected_num_ifos,
                    expected_state_size
                    + (expected_stream_size * expected_batch_size),
                ]
                assert config.output[0].dims == [
                    expected_batch_size,
                    expected_num_ifos,
                    expected_aframe_size,
                ]
            else:
                raise ValueError(f"Unexpected model {model.name} in repo")

        assert i == 3, f"Wrong number of models {i + 1}"

    return fn


# first set of tests will check that all the properties
# of the inputs to the neural network get set up properly
@pytest.fixture
def num_ifos():
    return 2


@pytest.fixture
def sample_rate():
    return 128


@pytest.fixture(params=[1, 4, 8])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[2, 4])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[8, 16])
def psd_length(request):
    return request.param


@pytest.fixture(params=[1, 2, 8])
def batch_size(request):
    return request.param


def test_export_for_shapes(
    repo_dir,
    batch_file,
    num_ifos,
    sample_rate,
    kernel_length,
    psd_length,
    batch_size,
    inference_sampling_rate,
    trace_network_weights,
    validate_repo,
    output_dir,
    platform,
):
    weights = output_dir / "weights.pt"
    trace_network_weights(
        num_ifos, weights, sample_rate, kernel_length, batch_size
    )

    # test fully from scratch behavior
    if (kernel_length * inference_sampling_rate) <= 1:
        context = pytest.raises(ValueError)
    else:
        context = nullcontext()

    fduration = 1
    expected_state_size = int(
        sample_rate
        * (
            kernel_length
            + fduration
            + psd_length
            - (1 / inference_sampling_rate)
        )
    )
    with context:
        export(
            weights,
            str(repo_dir),
            batch_file,
            num_ifos=num_ifos,
            inference_sampling_rate=inference_sampling_rate,
            kernel_length=kernel_length,
            psd_length=psd_length,
            sample_rate=sample_rate,
            batch_size=batch_size,
            fduration=fduration,
            streams_per_gpu=1,
            aframe_instances=1,
            platform=platform,
        )
        validate_repo(
            expected_aframe_instances=1,
            expected_snapshots=1,
            expected_versions=1,
            expected_num_ifos=num_ifos,
            expected_stream_size=int(sample_rate / inference_sampling_rate),
            expected_state_size=expected_state_size,
            expected_aframe_size=int(sample_rate * kernel_length),
            expected_crop=int(sample_rate * fduration),
            expected_batch_size=batch_size,
        )


def test_export_for_weights(
    repo_dir,
    output_dir,
    num_ifos,
    kernel_length,
    sample_rate,
    psd_length,
    inference_sampling_rate,
    fduration,
    trace_network_weights,
    validate_repo,
    batch_size,
    batch_file,
    platform,
):
    weights = output_dir / "weights.pt"
    trace_network_weights(
        num_ifos, weights, sample_rate, kernel_length, batch_size
    )

    expected_state_size = int(
        sample_rate
        * (
            kernel_length
            + fduration
            + psd_length
            - (1 / inference_sampling_rate)
        )
    )
    export(
        weights,
        str(repo_dir),
        batch_file,
        num_ifos=num_ifos,
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        psd_length=psd_length,
        batch_size=1,
        fduration=fduration,
        streams_per_gpu=1,
        aframe_instances=1,
        platform=platform,
    )
    validate_repo(
        expected_aframe_instances=1,
        expected_snapshots=1,
        expected_versions=1,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_aframe_size=int(sample_rate * kernel_length),
        expected_state_size=expected_state_size,
        expected_crop=int(sample_rate),
        expected_batch_size=1,
    )


# now test how different values of scaling parameters
# lead to different configs
@pytest.fixture(params=[None, 1])
def aframe_instances(request):
    return request.param


@pytest.fixture(params=[None, 1])
def preproc_instances(request):
    return request.param


@pytest.fixture(params=[1, 4])
def streams_per_gpu(request):
    return request.param


# indicates whether we ought to delete the contents
# of the model repository before doing export
@pytest.fixture(params=[True, False])
def clean(request):
    return request.param


def test_export_for_scaling(
    output_dir,
    repo_dir,
    batch_file,
    streams_per_gpu,
    aframe_instances,
    preproc_instances,
    clean,
    num_ifos,
    kernel_length,
    sample_rate,
    psd_length,
    inference_sampling_rate,
    fduration,
    validate_repo,
    trace_network_weights,
    batch_size,
    platform,
):
    weights = output_dir / "weights.pt"
    trace_network_weights(
        num_ifos, weights, sample_rate, kernel_length, batch_size
    )
    expected_state_size = int(
        sample_rate
        * (
            kernel_length
            + fduration
            + psd_length
            - (1 / inference_sampling_rate)
        )
    )

    if clean:
        p = repo_dir / "dummy_file.txt"
        p.write_text("dummy text")

    def run_export(
        aframe_instances=aframe_instances,
        preproc_instances=preproc_instances,
        clean=clean,
    ):
        export(
            weights,
            str(repo_dir),
            batch_file,
            num_ifos=num_ifos,
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            psd_length=psd_length,
            batch_size=1,
            fduration=1,
            streams_per_gpu=streams_per_gpu,
            aframe_instances=aframe_instances,
            clean=clean,
            platform=platform,
        )

    run_export()
    validate_repo(
        expected_aframe_instances=aframe_instances,
        expected_snapshots=streams_per_gpu,
        expected_versions=1,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_aframe_size=int(sample_rate * kernel_length),
        expected_state_size=expected_state_size,
        expected_crop=int(sample_rate),
        expected_batch_size=1,
    )

    # now check what happens if the repo already exists
    run_export()
    validate_repo(
        expected_aframe_instances=aframe_instances,
        expected_snapshots=streams_per_gpu,
        expected_versions=1 if clean else 2,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_aframe_size=int(sample_rate * kernel_length),
        expected_state_size=expected_state_size,
        expected_crop=int(sample_rate),
        expected_batch_size=1,
    )

    # now make sure if we change the scale
    # we get another version and the config changes
    run_export(
        aframe_instances=3, preproc_instances=preproc_instances, clean=False
    )
    validate_repo(
        expected_aframe_instances=3,
        expected_snapshots=streams_per_gpu,
        expected_versions=2 if clean else 3,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_aframe_size=int(sample_rate * kernel_length),
        expected_state_size=expected_state_size,
        expected_batch_size=1,
        expected_crop=int(sample_rate),
    )

    # now test to make sure an error gets raised if the
    # ensemble already exists but aframe is not part of it
    shutil.move(repo_dir / "aframe", repo_dir / "aaframe")
    aframe_config = repo_dir / "aaframe" / "config.pbtxt"
    config = aframe_config.read_text()
    config = re.sub('name: "aframe"', 'name: "aaframe"', config)
    aframe_config.write_text(config)

    ensemble_config = repo_dir / "aframe-stream" / "config.pbtxt"
    config = ensemble_config.read_text()
    config = re.sub('model_name: "aframe"', 'model_name: "aaframe"', config)
    ensemble_config.write_text(config)

    with pytest.raises(ValueError) as exc_info:
        run_export(clean=False)
    assert str(exc_info.value).endswith("model 'aframe'")

    # ensure that aframe got exported before things
    # went wrong with the ensemble. TODO: this is
    # actually probably undesirable behavior, but I'm
    # not sure the best way to handle it elegantly in
    # the export function. I guess a try-catch on the
    # ensemble section that deletes the most recent
    # aframe version if things go wrong?
    shutil.rmtree(repo_dir / "aaframe")
    validate_repo(
        expected_aframe_instances=aframe_instances,
        expected_snapshots=streams_per_gpu,
        expected_versions=(1, 3) if clean else (1, 4),
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_aframe_size=int(sample_rate * kernel_length),
        expected_state_size=expected_state_size,
        expected_batch_size=1,
        expected_crop=int(sample_rate),
    )
