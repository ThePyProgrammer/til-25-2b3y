from typing import Union

import torch.nn as nn

class ConvBlock(nn.Module):
    """A configurable convolutional block with optional batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = False,
        activation = nn.Tanh
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_batch_norm = use_batch_norm
        self.activation = activation()
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.use_batch_norm:
            x = self.bn(x)
        return x


class Conv3DBlock(nn.Module):
    """A configurable 3D convolutional block with optional batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]] = 3,
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 1,
        use_batch_norm: bool = False,
        activation = nn.ReLU
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_batch_norm = use_batch_norm
        self.activation = activation()
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.use_batch_norm:
            x = self.bn(x)
        return x
