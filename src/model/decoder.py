import torch
import torch.nn as nn
import torch.nn.functional as F
from .deform import DeformableConv2d
from einops import rearrange


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


GDT_INTER_CHANNELS = 16


class ImageProjection(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels: int = 64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        patches = rearrange(
            image,
            "b c (hg h) (wg w) -> b (c hg wg) h w",
            hg=image.shape[-2] // feature.shape[-2],
            wg=image.shape[-1] // feature.shape[-1],
        )
        patches = F.interpolate(
            patches, size=feature.shape[2:], mode="bilinear", align_corners=True
        )
        return self.conv_out(self.conv1(patches))


class Decoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        num_classes: int = 4,
        depth: bool = False,
    ) -> None:
        super().__init__()

        patch_channels = (3072, 768, 192, 48, 3)
        projection_channels = [channels[0] // 8, *(c // 8 for c in channels)]
        # Keep checkpoint names and seeded initialization order.
        for level, cin, cout in zip(
            range(5, 0, -1), patch_channels, projection_channels
        ):
            self.add_module(f"ipt_blk{level}", ImageProjection(cin, cout))

        # Zero initialization preserves the pretrained decoder before depth learns.
        self.dpt_blks = None
        if depth:
            self.dpt_blks = nn.ModuleList(
                ImageProjection(cin // 3, cout, inter_channels=16)
                for cin, cout in zip(patch_channels, projection_channels)
            )
            for block in self.dpt_blks:
                nn.init.zeros_(block.conv_out.weight)
                nn.init.zeros_(block.conv_out.bias)

        outputs = [*channels[1:], channels[-1] // 2]
        for level, cin, extra, cout in zip(
            range(4, 0, -1), channels, projection_channels, outputs
        ):
            self.add_module(f"decoder_block{level}", BasicDecBlk(cin + extra, cout))

        self.conv_out1 = nn.Sequential(
            nn.Conv2d(outputs[-1] + projection_channels[-1], num_classes, 1)
        )
        for level, cin, cout in zip(range(4, 1, -1), channels[1:], outputs):
            self.add_module(f"lateral_block{level}", BasicLatBlk(cin, cout))
        for level, cout in zip(range(4, 1, -1), outputs):
            self.add_module(f"conv_ms_spvn_{level}", nn.Conv2d(cout, num_classes, 1))
        for level, cout in zip(range(4, 1, -1), outputs):
            self.add_module(
                f"gdt_convs_{level}",
                nn.Sequential(
                    nn.Conv2d(cout, GDT_INTER_CHANNELS, 3, padding=1),
                    nn.BatchNorm2d(GDT_INTER_CHANNELS),
                    nn.ReLU(inplace=True),
                ),
            )
        for branch in ("pred", "attn"):
            for level in range(4, 1, -1):
                self.add_module(
                    f"gdt_convs_{branch}_{level}",
                    nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1)),
                )

    def guidance(self, logits: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        # All class interfaces contribute; background class 0 is not privileged.
        probs = logits.detach().softmax(dim=1)
        dx = F.pad(
            (probs[..., 1:] - probs[..., :-1]).abs().sum(1, keepdim=True), (0, 1)
        )
        dy = F.pad(
            (probs[..., 1:, :] - probs[..., :-1, :]).abs().sum(1, keepdim=True),
            (0, 0, 0, 1),
        )
        edge = F.interpolate(
            dx + dy, size=gradient.shape[2:], mode="bilinear", align_corners=True
        )
        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gradient = gradient.detach().abs()
        gradient = gradient / gradient.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (gradient * edge).detach()

    def _project(
        self,
        level: int,
        x: torch.Tensor,
        depth: torch.Tensor | None,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        patches = getattr(self, f"ipt_blk{5 - level}")(x, feature)
        if self.dpt_blks is not None:
            if depth is None:
                raise ValueError("This decoder was built with depth; pass a depth map")
            patches = patches + self.dpt_blks[level](depth, feature)
        return torch.cat((feature, patches), dim=1)

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        depth: torch.Tensor | None = None,
        auxiliary: bool = False,
    ) -> tuple[
        list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]] | None
    ]:
        if self.training and auxiliary:
            x, x1, x2, x3, x4, gradient = features
        else:
            x, x1, x2, x3, x4 = features
        side, gdt_preds, gdt_labels = [], [], []
        p = self._project(0, x, depth, x4)
        for level, skip in zip(range(4, 1, -1), (x3, x2, x1)):
            block = getattr(self, f"decoder_block{level}")
            head = getattr(self, f"conv_ms_spvn_{level}")
            gdt = getattr(self, f"gdt_convs_{level}")
            gdt_pred = getattr(self, f"gdt_convs_pred_{level}")
            gdt_attn = getattr(self, f"gdt_convs_attn_{level}")
            lateral = getattr(self, f"lateral_block{level}")
            p = block(p)
            p_gdt = gdt(p)
            if self.training:
                logits = head(p)
                side.append(logits)
                if auxiliary:
                    gdt_labels.append(self.guidance(logits, gradient))
                    gdt_preds.append(gdt_pred(p_gdt))
            p = p * gdt_attn(p_gdt).sigmoid()
            p = F.interpolate(
                p, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
            p = self._project(5 - level, x, depth, p + lateral(skip))

        p = self.decoder_block1(p)
        p = F.interpolate(p, size=x.shape[2:], mode="bilinear", align_corners=True)
        logits = [*side, self.conv_out1(self._project(4, x, depth, p))]
        guidance = (gdt_preds, gdt_labels) if self.training and auxiliary else None
        return logits, guidance
