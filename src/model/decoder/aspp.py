import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform import DeformableConv2d


class _ASPPModuleDeformable(nn.Module):
    def __init__(
        self, in_channels: int, planes: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels=in_channels,
            out_channels=planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        parallel_block_sizes: tuple[int, ...] = (1, 3, 7),
    ) -> None:
        super().__init__()
        out_channels = in_channels
        self.planes = 256

        self.aspp1 = _ASPPModuleDeformable(
            in_channels=in_channels,
            planes=self.planes,
            kernel_size=1,
            padding=0,
        )

        self.aspp_deforms = nn.ModuleList(
            [
                _ASPPModuleDeformable(
                    in_channels=in_channels,
                    planes=self.planes,
                    kernel_size=conv_size,
                    padding=int(conv_size // 2),
                )
                for conv_size in parallel_block_sizes
            ]
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(
            in_channels=self.planes * (2 + len(self.aspp_deforms)),
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(
            x5,
            size=x1.size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)
