import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .block import BasicDecBlk, BasicLatBlk

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
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        ipt_blk_in_channels = [3072, 768, 192, 48, 3]
        ipt_blk_out_channels = [channels[i] // 8 for i in range(4)]

        self.ipt_blk5 = ImageProjection(
            ipt_blk_in_channels[0], ipt_blk_out_channels[0], inter_channels=64
        )
        self.ipt_blk4 = ImageProjection(
            ipt_blk_in_channels[1], ipt_blk_out_channels[0], inter_channels=64
        )
        self.ipt_blk3 = ImageProjection(
            ipt_blk_in_channels[2], ipt_blk_out_channels[1], inter_channels=64
        )
        self.ipt_blk2 = ImageProjection(
            ipt_blk_in_channels[3], ipt_blk_out_channels[2], inter_channels=64
        )
        self.ipt_blk1 = ImageProjection(
            ipt_blk_in_channels[4], ipt_blk_out_channels[3], inter_channels=64
        )

        bb_neck_out_channels = channels.copy()

        dec_blk_out_channels = [c for c in bb_neck_out_channels[1:]] + [
            bb_neck_out_channels[-1] // 2
        ]

        dec_blk_in_channels = [
            bb_neck_out_channels[i] + ipt_blk_out_channels[max(0, i - 1)]
            for i in range(len(bb_neck_out_channels))
        ]

        self.decoder_block4 = BasicDecBlk(
            dec_blk_in_channels[0], dec_blk_out_channels[0]
        )
        self.decoder_block3 = BasicDecBlk(
            dec_blk_in_channels[1], dec_blk_out_channels[1]
        )
        self.decoder_block2 = BasicDecBlk(
            dec_blk_in_channels[2], dec_blk_out_channels[2]
        )
        self.decoder_block1 = BasicDecBlk(
            dec_blk_in_channels[3], dec_blk_out_channels[3]
        )

        conv_out1_in_channels = dec_blk_out_channels[3] + ipt_blk_out_channels[3]
        self.conv_out1 = nn.Sequential(nn.Conv2d(conv_out1_in_channels, num_classes, 1))

        self.lateral_block4 = BasicLatBlk(
            bb_neck_out_channels[1], dec_blk_out_channels[0]
        )
        self.lateral_block3 = BasicLatBlk(
            bb_neck_out_channels[2], dec_blk_out_channels[1]
        )
        self.lateral_block2 = BasicLatBlk(
            bb_neck_out_channels[3], dec_blk_out_channels[2]
        )

        self.conv_ms_spvn_4 = nn.Conv2d(dec_blk_out_channels[0], num_classes, 1)
        self.conv_ms_spvn_3 = nn.Conv2d(dec_blk_out_channels[1], num_classes, 1)
        self.conv_ms_spvn_2 = nn.Conv2d(dec_blk_out_channels[2], num_classes, 1)

        self.gdt_convs_4 = nn.Sequential(
            nn.Conv2d(dec_blk_out_channels[0], GDT_INTER_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(GDT_INTER_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_3 = nn.Sequential(
            nn.Conv2d(dec_blk_out_channels[1], GDT_INTER_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(GDT_INTER_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_2 = nn.Sequential(
            nn.Conv2d(dec_blk_out_channels[2], GDT_INTER_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(GDT_INTER_CHANNELS),
            nn.ReLU(inplace=True),
        )

        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))

        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))

    def guidance(self, logits: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return gradient * F.interpolate(
                logits, size=gradient.shape[2:], mode="bilinear", align_corners=True
            )
        # All class interfaces contribute; background class 0 is not privileged.
        probs = logits.detach().softmax(dim=1)
        dx = F.pad((probs[..., 1:] - probs[..., :-1]).abs().sum(1, keepdim=True), (0, 1))
        dy = F.pad((probs[..., 1:, :] - probs[..., :-1, :]).abs().sum(1, keepdim=True), (0, 0, 0, 1))
        edge = F.interpolate(dx + dy, size=gradient.shape[2:], mode="bilinear", align_corners=True)
        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gradient = gradient.detach().abs()
        gradient = gradient / gradient.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (gradient * edge).detach()

    def forward(
        self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> list[torch.Tensor] | tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        if self.training:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features

        outs = []

        x4 = torch.cat((x4, self.ipt_blk5(x, x4)), dim=1)

        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.training else None

        p4_gdt = self.gdt_convs_4(p4)
        if self.training:
            gdt_label_main_4 = self.guidance(m4, gdt_gt)
            outs_gdt_label.append(gdt_label_main_4)
            gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
            outs_gdt_pred.append(gdt_pred_4)
        gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
        p4 = p4 * gdt_attn_4

        _p4 = F.interpolate(
            p4,
            size=x3.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        _p3 = _p4 + self.lateral_block4(x3)

        _p3 = torch.cat((_p3, self.ipt_blk4(x, _p3)), dim=1)

        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.training else None

        p3_gdt = self.gdt_convs_3(p3)
        if self.training:
            gdt_label_main_3 = self.guidance(m3, gdt_gt)
            outs_gdt_label.append(gdt_label_main_3)
            gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
            outs_gdt_pred.append(gdt_pred_3)
        gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
        p3 = p3 * gdt_attn_3

        _p3 = F.interpolate(
            p3,
            size=x2.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        _p2 = _p3 + self.lateral_block3(x2)

        _p2 = torch.cat((_p2, self.ipt_blk3(x, _p2)), dim=1)

        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.training else None

        p2_gdt = self.gdt_convs_2(p2)
        if self.training:
            gdt_label_main_2 = self.guidance(m2, gdt_gt)
            outs_gdt_label.append(gdt_label_main_2)
            gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
            outs_gdt_pred.append(gdt_pred_2)
        gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
        p2 = p2 * gdt_attn_2

        _p2 = F.interpolate(
            p2,
            size=x1.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        _p1 = _p2 + self.lateral_block2(x1)

        _p1 = torch.cat((_p1, self.ipt_blk2(x, _p1)), dim=1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(
            _p1,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        _p1 = torch.cat((_p1, self.ipt_blk1(x, _p1)), dim=1)

        p1_out = self.conv_out1(_p1)

        if self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)

        if not self.training:
            return outs

        return [outs_gdt_pred, outs_gdt_label], outs
