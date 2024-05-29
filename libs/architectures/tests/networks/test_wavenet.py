import torch
from architectures.networks import WaveNet


def test_wavenet():
    in_channels = 2
    arch = WaveNet(
        in_channels,
        res_channels=4,
        layers_per_block=2,
        num_blocks=2,
        kernel_size=2,
    )

    data = torch.randn(512, in_channels, 1024)
    x = arch(data)
    assert x.shape == (512, 1)
