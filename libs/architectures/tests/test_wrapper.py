import sys
from typing import Callable

from torch import nn

from bbhnet.architectures import architecturize
from bbhnet.architectures.wrapper import architectures


def set_argv(*args):
    sys.argv = [None] + list(args)


def test_resnet_wrappers():
    def func(architecture: Callable, learning_rate: float):
        nn = architecture(2)

        # arch will be defined in the dict loop later
        assert isinstance(nn, arch)

        return nn.residual_layers, learning_rate

    wrapped = architecturize(func)

    # now try to use this wrapped function at
    # the "command" line for both architectures
    for name, arch in architectures.items():
        if name not in ("resnet", "bottleneck"):
            continue

        set_argv(
            "--learning-rate",
            "1e-3",
            name,
            "--layers",
            "2",
            "2",
            "2",
            "--kernel-size",
            "7",
        )
        layers, lr = wrapped()

        # make sure the parameters that got passed are correct
        assert lr == 1e-3
        assert len(layers) == 3

        # iterate through each residual layer and
        # make sure things look right
        for layer in layers:
            for block in layer:
                # make sure all the blocks are the right type
                # (i.e. we have the correct architecture)
                assert isinstance(block, arch.block)

                # make sure all the convolutional kernels
                # have the appropriate shape (1 is allowed
                # because that's what we use when downsampling)
                for module in block.modules():
                    if isinstance(module, nn.Conv1d):
                        assert module.weight.shape[-1] in (1, 7)
