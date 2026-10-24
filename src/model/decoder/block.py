import torch
import torch.nn as nn

from .aspp import ASPPDeformable


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        inter_channels = 64

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)

        self.conv_out = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
