import re
import shutil
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from export import export
from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

from bbhnet.architectures import Preprocessor, ResNet


# set up a directory for the entirety of the session
# which will store all the weight values of each
# NN we need to create in response to a particular
# num_ifos/sample_rate/kernel_length combination
@pytest.fixture(scope="session")
def weights_dir():
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    yield weights_dir
    shutil.rmtree(weights_dir)


# only create a new neural network if the weights for
# a network of this num_ifos/sample_rate/kernel_length
# combination has not yet been created. Otherwise just
# return the path to those weights as-is
@pytest.fixture
def architecture():
    return lambda num_ifos: ResNet(num_ifos, [2, 2])


@pytest.fixture
def get_network_weights(weights_dir, architecture):
    def fn(num_ifos, sample_rate, kernel_length, target):
        weights = weights_dir / f"{num_ifos}-{sample_rate}-{kernel_length}.pt"
        if not weights.exists():
            preprocessor = Preprocessor(num_ifos, sample_rate, kernel_length)
            bbhnet = architecture(num_ifos)
            model = torch.nn.Sequential(preprocessor, bbhnet)
            torch.save(model.state_dict(prefix=""), weights)

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


def load_config(config_path: Path):
    config = ModelConfig()
    text_format.Merge(config_path.read_text(), config)
    return config


@pytest.fixture
def validate_repo(repo_dir):
    def fn(
        expected_instances,
        expected_snapshots,
        expected_versions,
        expected_num_ifos,
        expected_stream_size,
        expected_kernel_size,
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

                assert config.input[0].dims == [
                    1,
                    expected_num_ifos,
                    expected_stream_size * expected_batch_size,
                ]
                assert config.output[0].dims == [
                    expected_batch_size,
                    expected_num_ifos,
                    expected_kernel_size,
                ]

                assert (model / "1" / "model.onnx").exists()
                assert not (model / "2").is_dir()
            elif model.name == "bbhnet":
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
                            "Didn't expect bbhnet instances but found "
                            f"instance group {instance_group}"
                        )
                    assert instance_group.count == expected_instances

                assert config.input[0].dims == [
                    expected_batch_size,
                    expected_num_ifos,
                    expected_kernel_size,
                ]
                assert config.output[0].dims == [expected_batch_size, 1]

                for j in range(expected_versions):
                    assert (model / str(j + 1) / "model.onnx").is_file()
                assert not (model / str(j + 2)).is_dir()

                assert config.optimization.graph.level == -1
            elif model.name == "bbhnet-stream":
                assert (model / "1").is_dir()
                assert not (model / "2").is_dir()
            else:
                raise ValueError(f"Unexpected model {model.name} in repo")

        assert i == 2, f"Wrong number of models {i + 1}"

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


@pytest.fixture(params=[1, 2])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[1, 2, 8])
def batch_size(request):
    return request.param


def test_export_for_shapes(
    repo_dir,
    output_dir,
    num_ifos,
    sample_rate,
    kernel_length,
    batch_size,
    inference_sampling_rate,
    architecture,
    get_network_weights,
    validate_repo,
):
    weights = output_dir / "weights.pt"
    get_network_weights(num_ifos, sample_rate, kernel_length, weights)

    # test fully from scratch behavior
    if (kernel_length * inference_sampling_rate) <= 1:
        context = pytest.raises(ValueError)
    else:
        context = nullcontext()

    with context:
        export(
            architecture,
            str(repo_dir),
            output_dir,
            num_ifos=num_ifos,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            batch_size=batch_size,
            weights=weights,
            streams_per_gpu=1,
            instances=1,
        )
        validate_repo(
            expected_instances=1,
            expected_snapshots=1,
            expected_versions=1,
            expected_num_ifos=num_ifos,
            expected_stream_size=int(sample_rate / inference_sampling_rate),
            expected_kernel_size=int(sample_rate * kernel_length),
            expected_batch_size=batch_size,
        )


# next test how passing different values of the
# `weights` parameter causes different behavior.
# - `None` will indicate to not pass anything to the function,
#       which it will use to infer that it should look for
#       a `weights.pt` file in the `output_dir`
# - `None` will indicate to pass output_dir to `weights, which
#       will indicate that this is a directory that it ought
#       to check for a `weights.pt`
# - `weights.pt` indicates a full, sensible path to a weights file
# - `other.pdf` just tests that even with a weird name, the
#       path is still resolved appropriately
@pytest.mark.parametrize("weights", [None, "", "weights.pt", "other.pdf"])
def test_export_for_weights(
    repo_dir,
    output_dir,
    weights,
    architecture,
    get_network_weights,
    validate_repo,
):
    num_ifos = 2
    kernel_length = 1
    sample_rate = 128
    inference_sampling_rate = 4

    if not weights:
        target = output_dir / "weights.pt"
        if weights is None:
            weights = output_dir
        else:
            weights = output_dir / "weights.pt"
    else:
        weights = target = output_dir / weights
    get_network_weights(num_ifos, sample_rate, kernel_length, target)

    export(
        architecture,
        str(repo_dir),
        output_dir,
        num_ifos=num_ifos,
        kernel_length=kernel_length,
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        batch_size=1,
        weights=weights,
        streams_per_gpu=1,
        instances=1,
    )
    validate_repo(
        expected_instances=1,
        expected_snapshots=1,
        expected_versions=1,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_kernel_size=int(sample_rate * kernel_length),
        expected_batch_size=1,
    )


# now test how different values of scaling parameters
# lead to different configs
@pytest.fixture(params=[None, 1])
def instances(request):
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
    repo_dir,
    output_dir,
    streams_per_gpu,
    instances,
    clean,
    architecture,
    validate_repo,
    get_network_weights,
):
    num_ifos = 2
    kernel_length = 1
    sample_rate = 128
    inference_sampling_rate = 4

    weights = output_dir / "weights.pt"
    get_network_weights(num_ifos, sample_rate, kernel_length, weights)

    if clean:
        p = repo_dir / "dummy_file.txt"
        p.write_text("dummy text")

    def run_export(instances=instances, clean=clean):
        export(
            architecture,
            str(repo_dir),
            output_dir,
            num_ifos=num_ifos,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            batch_size=1,
            weights=weights,
            streams_per_gpu=streams_per_gpu,
            instances=instances,
            clean=clean,
        )

    run_export()
    validate_repo(
        expected_instances=instances,
        expected_snapshots=streams_per_gpu,
        expected_versions=1,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_kernel_size=int(sample_rate * kernel_length),
        expected_batch_size=1,
    )

    # now check what happens if the repo already exists
    run_export()
    validate_repo(
        expected_instances=instances,
        expected_snapshots=streams_per_gpu,
        expected_versions=1 if clean else 2,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_kernel_size=int(sample_rate * kernel_length),
        expected_batch_size=1,
    )

    # now make sure if we change the scale
    # we get another version and the config changes
    run_export(instances=3, clean=False)
    validate_repo(
        expected_instances=3,
        expected_snapshots=streams_per_gpu,
        expected_versions=2 if clean else 3,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_kernel_size=int(sample_rate * kernel_length),
        expected_batch_size=1,
    )

    # now test to make sure an error gets raised if the
    # ensemble already exists but bbhnet is not part of it
    shutil.move(repo_dir / "bbhnet", repo_dir / "bbbhnet")
    bbhnet_config = repo_dir / "bbbhnet" / "config.pbtxt"
    config = bbhnet_config.read_text()
    config = re.sub('name: "bbhnet"', 'name: "bbbhnet"', config)
    bbhnet_config.write_text(config)

    ensemble_config = repo_dir / "bbhnet-stream" / "config.pbtxt"
    config = ensemble_config.read_text()
    config = re.sub('model_name: "bbhnet"', 'model_name: "bbbhnet"', config)
    ensemble_config.write_text(config)

    with pytest.raises(ValueError) as exc_info:
        run_export(clean=False)
    assert str(exc_info.value).endswith("model 'bbhnet'")

    # ensure that bbhnet got exported before things
    # went wrong with thet ensemble. TODO: this is
    # actually probably undesirable behavior, but I'm
    # not sure the best way to handle it elegantly in
    # the export function. I guess a try-catch on the
    # ensemble section that deletes the most recent
    # bbhnet version if things go wrong?
    shutil.rmtree(repo_dir / "bbbhnet")
    validate_repo(
        expected_instances=instances,
        expected_snapshots=streams_per_gpu,
        expected_versions=1,
        expected_num_ifos=num_ifos,
        expected_stream_size=int(sample_rate / inference_sampling_rate),
        expected_kernel_size=int(sample_rate * kernel_length),
        expected_batch_size=1,
    )
