import sys
from typing import Callable
from unittest.mock import Mock

import torch

from aframe.architectures import architecturize
from aframe.architectures.wrapper import architectures


def set_argv(*args):
    sys.argv = [None] + list(args)


def test_resnet_wrappers():
    mock = Mock()

    def func(architecture: Callable, learning_rate: float):
        nn = architecture(2)

        # arch will be defined in the dict loop later
        assert isinstance(nn, arch)
        assert len(nn.residual_layers) == 3
        assert learning_rate == 1e-3

        # iterate through each residual layer and
        # make sure things look right
        for layer in nn.residual_layers:
            for block in layer:
                # make sure all the blocks are the right type
                # (i.e. we have the correct architecture)
                assert isinstance(block, arch.block)

                # make sure all the convolutional kernels
                # have the appropriate shape (1 is allowed
                # because that's what we use when downsampling)
                for module in block.modules():
                    if isinstance(module, torch.nn.Conv1d):
                        assert module.weight.shape[-1] in (1, 7)

        # call this at end to make sure all the
        # above code got called outside
        mock()

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
        wrapped()
        mock.assert_called()
