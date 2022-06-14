import re
import shutil

import pytest
import torch
from export import export

from bbhnet.architectures import ResNet
from bbhnet.data.transforms import WhiteningTransform


class AddOne(torch.nn.Module):
    def __init__(self, num_ifos: int) -> None:
        super().__init__()
        self.num_ifos = num_ifos

    def forward(self, x):
        return x + 1


@pytest.fixture(params=[128, 512])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 4, 8])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[0.5, 1, 2])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[None, 1, 2])
def instances(request):
    return request.param


@pytest.fixture(params=[1, 4])
def streams_per_gpu(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_ifos(request):
    return request.param


@pytest.fixture
def repo_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture(params=[None, "", "weights.pt", "other.pdf"])
def weights(request):
    return request.param


@pytest.fixture(params=[True, False])
def clean(request):
    return request.param


@pytest.fixture
def output_dir(tmp_path, num_ifos, sample_rate, kernel_length, weights):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    weights = weights or "weights.pt"
    preprocessor = WhiteningTransform(num_ifos, sample_rate, kernel_length)
    bbhnet = ResNet(num_ifos, [2, 2])
    model = torch.nn.Sequential(preprocessor, bbhnet)
    torch.save(model.state_dict(prefix=""), output_dir / weights)
    return output_dir


def test_export(
    repo_dir,
    output_dir,
    num_ifos,
    kernel_length,
    inference_sampling_rate,
    sample_rate,
    streams_per_gpu,
    instances,
    weights,
    clean,
):
    if weights == "":
        weights = output_dir
    elif weights is not None:
        weights = output_dir / weights

    def validate_repo(instances, versions):
        models = [i.name for i in repo_dir.iterdir()]
        assert len(models) == 3
        assert set(models) == set(["bbhnet", "snapshotter", "bbhnet-stream"])

        # verify that the instance group scale is correct
        bbhnet_config = repo_dir / "bbhnet" / "config.pbtxt"
        assert bbhnet_config.exists()
        config = bbhnet_config.read_text()
        has_scale = f"count: {instances}" in config
        assert has_scale ^ (instances is None)

        # TODO: check shapes in config
        # is this our business or do we trust quiver to test
        # for this correctly? I guess it's more of a test as
        # to whether we fed the arguments to quiver correctly

        # verify that we have all the versions we expect of bbhnet
        bbhnet_versions = list((repo_dir / "bbhnet").iterdir())
        bbhnet_versions = [i.name for i in bbhnet_versions]
        for i in range(1, versions + 1):
            assert str(i) in bbhnet_versions
            assert (repo_dir / "bbhnet" / str(i) / "model.onnx").is_file()

        # make sure we only ever have one snapshotter
        # and ensemble model version
        assert len(list((repo_dir / "snapshotter").iterdir())) == 2
        assert len(list((repo_dir / "bbhnet-stream").iterdir())) == 2

        # TODO: check shapes in configs

    def run_export(instances=instances, clean=clean):
        export(
            lambda num_ifos: ResNet(num_ifos, [2, 2]),
            str(repo_dir),
            output_dir,
            num_ifos=num_ifos,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            weights=weights,
            streams_per_gpu=streams_per_gpu,
            instances=instances,
            clean=clean,
        )

    # test fully from scratch behavior
    if kernel_length < (1 / inference_sampling_rate):
        with pytest.raises(ValueError):
            run_export()
        return

    run_export()
    validate_repo(instances, 1)

    # now check what happens if the repo already exists
    run_export()
    validate_repo(instances, 1 if clean else 2)

    # now make sure if we change the scale
    # we get another version and the config changes
    run_export(instances=3, clean=False)
    validate_repo(3, 2 if clean else 3)

    # now test to make sure an error gets raised if
    # the ensemble already exists but bbhnet is not
    # part of it
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
    validate_repo(None, 1)
