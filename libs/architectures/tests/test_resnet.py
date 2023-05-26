import h5py  # noqa
import pytest
import torch

from aframe.architectures.resnet import (
    BasicBlock,
    Bottleneck,
    BottleneckResNet,
    ChannelNorm,
    ResNet,
    conv1,
)


@pytest.fixture(params=[3, 7, 8, 11])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def stride(request):
    return request.param


@pytest.fixture(params=[2, 4])
def inplanes(request):
    return request.param


@pytest.fixture(params=[BasicBlock, Bottleneck])
def block(request):
    return request.param


def test_blocks(block, kernel_size, stride, sample_rate, inplanes):
    # TODO: test dilation for bottleneck
    planes = 4

    if stride > 1 or inplanes != planes * block.expansion:
        downsample = conv1(inplanes, planes * block.expansion, stride)
    else:
        downsample = None

    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            block = block(
                inplanes, planes, kernel_size, stride, downsample=downsample
            )
        return

    block = block(inplanes, planes, kernel_size, stride, downsample=downsample)
    x = torch.randn(8, inplanes, sample_rate)
    y = block(x)

    assert len(y.shape) == 3
    assert y.shape[1] == planes * block.expansion
    assert y.shape[2] == sample_rate // stride


@pytest.fixture(params=[1, 2, 3])
def num_ifos(request):
    return request.param


@pytest.fixture(params=[[2, 2, 2, 2], [2, 4, 4], [3, 4, 6, 3]])
def layers(request):
    return request.param


@pytest.fixture(params=[None, "stride", "dilation"])
def stride_type(request):
    return request.param


@pytest.fixture(params=[BottleneckResNet, ResNet])
def architecture(request):
    return request.param


def test_resnet(
    architecture, kernel_size, layers, num_ifos, sample_rate, stride_type
):
    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            nn = ResNet(num_ifos, layers, kernel_size)
        return

    if stride_type is not None:
        stride_type = [stride_type] * (len(layers) - 1)

    if (
        stride_type is not None
        and stride_type[0] == "dilation"
        and architecture == ResNet
    ):
        with pytest.raises(NotImplementedError):
            nn = architecture(
                num_ifos, layers, kernel_size, stride_type=stride_type
            )
        return

    nn = architecture(num_ifos, layers, kernel_size, stride_type=stride_type)
    x = torch.randn(8, num_ifos, sample_rate)
    y = nn(x)
    assert y.shape == (8, 1)

    with pytest.raises(ValueError):
        stride_type = ["stride"] * len(layers)
        nn = architecture(
            num_ifos, layers, kernel_size, stride_type=stride_type
        )
    with pytest.raises(ValueError):
        stride_type = ["strife"] * (len(layers) - 1)
        nn = architecture(
            num_ifos, layers, kernel_size, stride_type=stride_type
        )


@pytest.fixture(params=[None, 2])
def num_groups(request):
    return request.param


@pytest.fixture(params=[2, 4])
def num_channels(request):
    return request.param


def test_channel_norm(num_groups, num_channels):
    with pytest.raises(ValueError):
        ChannelNorm(num_channels, 3)

    norm = ChannelNorm(num_channels, num_groups)
    assert norm.num_groups == (num_groups or num_channels)

    # update the norm layers weights so that
    # we have something interesting to compare
    optim = torch.optim.SGD(norm.parameters(), lr=1e-1)
    for i in range(10):
        optim.zero_grad()
        x, y = [torch.randn(8, num_channels, 128) for _ in range(2)]
        y = 0.2 + 0.5 * y
        y_hat = norm(x)
        loss = ((y_hat - y) ** 2).mean()
        loss.backward()
        optim.step()

    # copy learned parameters into normal groupnorm
    # and verify that outputs are similar
    ref = torch.nn.GroupNorm(norm.num_groups, norm.num_channels)
    ref.weight.requires_grad = False
    ref.bias.requires_grad = False
    ref.weight.copy_(norm.weight.data[:, 0])
    ref.bias.copy_(norm.bias.data[:, 0])

    x = torch.randn(1024, num_channels, 128)
    y_ref = ref(x)
    y = norm(x)

    close = torch.isclose(y, y_ref, rtol=1e-6)
    num_wrong = (~close).sum()
    assert (num_wrong / y.numel()) < 0.01
