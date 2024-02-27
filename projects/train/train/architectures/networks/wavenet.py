import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.gated = GatedActivation()
        self.dilated = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.conv_res = nn.Conv1d(in_channels, in_channels, 1)
        self.skip_res = nn.Conv1d(in_channels, skip_channels, 1)

    def forward(self, x):
        identity = x

        # dilate, gate, and send through residual conv
        dilated = self.dilated(x)
        gated = self.gated(dilated)
        output = self.conv_res(gated)

        # add input to residual
        output += identity

        # perform skip convolution
        skip = self.skip_res(gated)

        return output, skip


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
        self, in_channels, res_channels, layers_per_block: int, num_blocks: int
    ):
        super().__init__()
        self.init_conv = torch.nn.Conv1d(in_channels, res_channels, 1)
        self.layers_per_block = layers_per_block
        self.num_blocks = num_blocks
        self.res_channels = res_channels
        self.in_channels = in_channels
        self.dense = DenseNet(res_channels)
        self.blocks = self.build_blocks()

    @property
    def receptive_field(self):
        layers = [
            2**i for i in range(0, self.layers_per_block)
        ] * self.num_blocks
        layers = torch.tensor(layers)
        return int(torch.sum(layers))

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
        for d in self.dilations:
            blocks.append(
                WavenetBlock(
                    self.res_channels,
                    self.res_channels,
                    kernel_size=2,
                    dilation=d,
                )
            )
        return nn.ModuleList(blocks)

    def forward(self, x):
        x = self.init_conv(x)
        output = torch.zeros(x.size(), device=x.device)
        for i, block in enumerate(self.blocks):
            # output is the next input
            x, skip = block(x)
            output += skip
        output = self.dense(output)
        return output
