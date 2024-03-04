from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from ml4gw.nn.norm import GroupNorm1DGetter, NormLayer


class GatedActivation(torch.nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        one = self.tanh(x)
        two = self.sig(x)
        return one * two


class WavenetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        dilation: int,
        kernel_size: int = 2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last: bool = False,
    ):
        super().__init__()
        self.last = last
        self.gated = GatedActivation()
        self.dilated = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm1 = norm_layer(in_channels)
        if not last:
            self.conv_res = (
                nn.Conv1d(in_channels, in_channels, 1) if not last else None
            )
            self.norm2 = norm_layer(in_channels)
        self.skip_res = nn.Conv1d(in_channels, skip_channels, 1)

    def forward(self, x, skip_size):
        # dilate, gate, and send through residual conv
        dilated = self.dilated(x)
        norm = self.norm1(dilated)
        gated = self.gated(norm)

        res = None
        if not self.last:
            res = self.conv_res(gated)
            res = self.norm2(res)
            # add input to residual
            res += x[..., -res.size(-1) :]

        # perform skip convolution
        skip = self.skip_res(gated)
        skip = skip[..., -skip_size:]
        return res, skip


class DenseNet(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)
        self.relu = torch.nn.ReLU()
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(channels, 1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.adaptive_pool(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)

        return output


class WaveNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        res_channels,
        layers_per_block: int,
        num_blocks: int,
        kernel_size: int = 2,
        norm_layer: Optional[NormLayer] = None,
    ):
        self.norm_layer = norm_layer or GroupNorm1DGetter()
        super().__init__()
        self.init_conv = nn.Conv1d(
            in_channels, res_channels, kernel_size=2, dilation=1
        )
        self.norm1 = norm_layer(res_channels)
        self.layers_per_block = layers_per_block
        self.num_blocks = num_blocks
        self.res_channels = res_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dense = DenseNet(res_channels)
        self.blocks = self.build_blocks()
        self.receptive_field = self.calc_receptive_field()

    def calc_receptive_field(self):
        return np.sum(
            [(2**i) for i in range(self.layers_per_block)] * self.num_blocks
        )

    def output_size(self, x):
        size = int(x.size(-1)) - self.receptive_field
        return size

    @property
    def dilations(self):
        return [
            2**i for i in range(0, self.layers_per_block)
        ] * self.num_blocks

    def build_blocks(self):
        blocks = []
        for i, d in enumerate(self.dilations):
            last = i == len(self.dilations) - 1
            blocks.append(
                WavenetBlock(
                    self.res_channels,
                    self.res_channels,
                    kernel_size=self.kernel_size,
                    dilation=d,
                    norm_layer=self.norm_layer,
                    last=last,
                )
            )
        return nn.ModuleList(blocks)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.norm1(x)
        output_size = self.output_size(x)
        size = (x.size(0), self.res_channels, output_size)
        output = torch.zeros(size, device=x.device)
        for block in self.blocks:
            # output is the next input
            x, skip = block(x, output_size)
            output += skip

        output = self.dense(output)
        return output
