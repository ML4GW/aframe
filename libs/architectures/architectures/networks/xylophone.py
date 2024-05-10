from collections import namedtuple
from typing import Optional

import torch

from ml4gw.nn.resnet.resnet_1d import GroupNorm1DGetter, NormLayer, convN


class XylophoneResidualBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__()
        self.conv1 = convN(inplanes, planes, kernel_size, stride, 1, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = convN(planes, planes, kernel_size, 1, 1, dilation)
        self.bn2 = norm_layer(planes)
        if stride > 1:
            self.downsample = torch.nn.Conv1d(
                inplanes,
                planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.downsample = None

        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def get_pad(self, size):
        padding = self.dilation * int(self.kernel_size // 2)
        outsize = size + 2 * padding - self.dilation * (self.kernel_size - 1)
        outsize = int((outsize - 1) // self.stride + 1)

        downsize = (outsize - 1) * self.stride + 1
        diff = size - downsize
        if not diff:
            return None, None
        left = int(diff // 2)
        right = diff - left
        return left, -right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # if we're downsampling on this layer, figure out
            # how much of x we need to chop off the ends in
            # order to get something that has the correct shape
            # to apply as a residual
            left, right = self.get_pad(x.size(-1))
            x = x[:, :, left:right]
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Xylophone(torch.nn.Module):
    block = XylophoneResidualBlock

    def __init__(
        self,
        num_ifos: int,
        classes: int,
        norm_layer: Optional[NormLayer] = None,
        base_channels: int = 8,
    ):
        super().__init__()
        self._norm_layer = norm_layer or GroupNorm1DGetter()

        self.base_channels = base_channels
        self.initial = torch.nn.Sequential(
            torch.nn.Conv1d(
                num_ifos,
                self.base_channels,
                kernel_size=7,
                stride=2,
                groups=1,
                padding=3,
            ),
            self._norm_layer(self.base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.out_channels = 0
        self.octaves = torch.nn.ModuleDict()

        # not sure how to make this more configurable,
        # but at least here's something to help make
        # the organization of the octaves a bit more neat
        Octave = namedtuple(
            "Octave", ["base_channels", "stride", "dilation", "layers"]
        )

        # organize towers of short residual layers we'll
        # call octaves, each one with its own level of
        # dilation. The true stride is stride * dilation,
        # so shorter dilations will have longer strides in
        # order to reduce the amount of compute in that tower.
        # However, the field of view of these octaves will grow
        # less quickly, so we'll need to use more layers. For that
        # reason, we'll also reduce the number of channels when
        # we use more layers to balance the overall compute-per-octave
        octaves = [
            Octave(64, stride=1, dilation=8, layers=[2, 2]),
            Octave(32, stride=2, dilation=4, layers=[2, 2, 2]),
            Octave(16, stride=3, dilation=2, layers=[2, 2, 2, 2]),
            Octave(8, stride=3, dilation=1, layers=[2, 4, 3, 2]),
        ]
        for i, octave in enumerate(octaves):
            octave = self.make_octave(
                octave.base_channels,
                kernel_size=7,
                stride=octave.stride,
                dilation=octave.dilation,
                layers=octave.layers,
            )
            self.octaves[f"octave{i}"] = octave

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(self.out_channels, classes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

            if isinstance(m, XylophoneResidualBlock):
                torch.nn.init.constant_(m.bn2.weight, 0)

    def make_octave(
        self,
        base_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        layers: list[int],
    ):
        inplanes = self.base_channels
        octave = []
        for i, num_blocks in enumerate(layers):
            # each layer in our octave will consist of
            # multiple residual blocks. Only on the first
            # block within each layer will we downsample
            # along the time dimension by a factor of
            # stride and upsample along the channel dimension
            # by a factor of 2
            outplanes = base_channels * 2**i
            layer = []
            for j in range(num_blocks):
                block = XylophoneResidualBlock(
                    inplanes=outplanes if j else inplanes,
                    planes=outplanes,
                    kernel_size=kernel_size,
                    stride=1 if j else stride * dilation,
                    dilation=dilation,
                    norm_layer=self._norm_layer,
                )
                layer.append(block)
            inplanes = outplanes
            octave.append(torch.nn.Sequential(*layer))

        self.out_channels += outplanes
        return torch.nn.Sequential(*octave)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)

        # break up x into equal-sized, half-overlapping
        # segments to apply each octave network to
        num_octaves = len(self.octaves)
        size = x.size(-1)
        octave_size = int(2 * size // num_octaves)
        octave_step = int(octave_size // 2)
        outputs = []
        for i in range(num_octaves):
            start = i * octave_step
            xoct = x[:, :, start : start + octave_size]
            xoct = self.octaves[f"octave{i}"](xoct)
            xoct = self.avgpool(xoct)
            outputs.append(xoct)

        # concatenate the octave outputs along their
        # channel dimension and then run them through
        # a fully connected layer
        x = torch.cat(outputs, axis=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
