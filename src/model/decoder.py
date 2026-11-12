import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .deform import DeformableConv2d

GDT_CHANNELS = 16


def resize_bilinear(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ASPPBranch(nn.Module):
    def __init__(self, in_channels, planes, kernel_size):
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, channels, planes=256):
        super().__init__()
        self.aspp1 = ASPPBranch(channels, planes, 1)
        self.aspp_deforms = nn.ModuleList(
            ASPPBranch(channels, planes, size) for size in (1, 3, 7)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(planes * 5, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        branches = [self.aspp1(x), *(branch(x) for branch in self.aspp_deforms)]
        pooled = resize_bilinear(self.global_avg_pool(x), branches[0].shape[2:])
        x = self.conv1(torch.cat((*branches, pooled), dim=1))
        return self.dropout(F.relu(self.bn1(x)))


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.dec_att = ASPPDeformable(inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        return self.bn_out(self.conv_out(self.dec_att(x)))


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class ImageProjection(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)

    def forward(self, image, feature):
        patches = rearrange(
            image,
            "b c (hg h) (wg w) -> b (c hg wg) h w",
            hg=image.shape[-2] // feature.shape[-2],
            wg=image.shape[-1] // feature.shape[-1],
        )
        patches = resize_bilinear(patches, feature.shape[2:])
        return self.conv_out(self.conv1(patches))


def make_guidance(logits, gradient):
    probs = logits.detach().softmax(dim=1)
    dx = F.pad((probs[..., 1:] - probs[..., :-1]).abs().sum(1, keepdim=True), (0, 1))
    dy = F.pad(
        (probs[..., 1:, :] - probs[..., :-1, :]).abs().sum(1, keepdim=True),
        (0, 0, 0, 1),
    )
    edge = resize_bilinear(dx + dy, gradient.shape[2:])
    edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    gradient = gradient.detach().abs()
    gradient = gradient / gradient.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return gradient * edge


class Decoder(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        patch_channels = (3072, 768, 192, 48, 3)
        projections = [channels[0] // 8, *(c // 8 for c in channels)]
        for level, cin, cout in zip(range(5, 0, -1), patch_channels, projections):
            self.add_module(f"ipt_blk{level}", ImageProjection(cin, cout))
        outputs = [*channels[1:], channels[-1] // 2]
        for level, cin, extra, cout in zip(
            range(4, 0, -1), channels, projections, outputs
        ):
            self.add_module(f"decoder_block{level}", BasicDecBlk(cin + extra, cout))
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(outputs[-1] + projections[-1], num_classes, 1)
        )
        for level, cin, cout in zip(range(4, 1, -1), channels[1:], outputs):
            self.add_module(f"lateral_block{level}", BasicLatBlk(cin, cout))
        for level, cout in zip(range(4, 1, -1), outputs):
            self.add_module(f"conv_ms_spvn_{level}", nn.Conv2d(cout, num_classes, 1))
        for level, cout in zip(range(4, 1, -1), outputs):
            self.add_module(
                f"gdt_convs_{level}",
                nn.Sequential(
                    nn.Conv2d(cout, GDT_CHANNELS, 3, padding=1),
                    nn.BatchNorm2d(GDT_CHANNELS),
                    nn.ReLU(inplace=True),
                ),
            )
        for branch in ("pred", "attn"):
            for level in range(4, 1, -1):
                self.add_module(
                    f"gdt_convs_{branch}_{level}",
                    nn.Sequential(nn.Conv2d(GDT_CHANNELS, 1, 1)),
                )

    def project(self, level, x, feature):
        patches = getattr(self, f"ipt_blk{5 - level}")(x, feature)
        return torch.cat((feature, patches), dim=1)

    def forward(self, features, gradient=None):
        x, x1, x2, x3, x4 = features
        side, gdt_preds, gdt_labels = [], [], []
        p = self.project(0, x, x4)
        for level, skip in zip(range(4, 1, -1), (x3, x2, x1)):
            p = getattr(self, f"decoder_block{level}")(p)
            p_gdt = getattr(self, f"gdt_convs_{level}")(p)
            if self.training:
                logits = getattr(self, f"conv_ms_spvn_{level}")(p)
                side.append(logits)
                if gradient is not None:
                    gdt_labels.append(make_guidance(logits, gradient))
                    gdt_preds.append(getattr(self, f"gdt_convs_pred_{level}")(p_gdt))
            p = p * getattr(self, f"gdt_convs_attn_{level}")(p_gdt).sigmoid()
            p = resize_bilinear(p, skip.shape[2:])
            lateral = getattr(self, f"lateral_block{level}")(skip)
            p = self.project(5 - level, x, p + lateral)
        p = resize_bilinear(self.decoder_block1(p), x.shape[2:])
        logits = [*side, self.conv_out1(self.project(4, x, p))]
        guidance = (gdt_preds, gdt_labels) if gradient is not None else None
        return logits, guidance
