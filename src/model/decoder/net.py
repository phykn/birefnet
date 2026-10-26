import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .block import BasicDecBlk, BasicLatBlk

GDT_INTER_CHANNELS = 16


def run_block(block: nn.Module, x: torch.Tensor, recompute: bool) -> torch.Tensor:
    """Call a block, optionally dropping its activations and recomputing them in backward."""
    if recompute and torch.is_grad_enabled():
        return checkpoint(block, x, use_reentrant=False)
    return block(x)


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
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.grad_checkpoint = grad_checkpoint

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

        # Depth enters next to the image patches at every level. The last conv of each
        # branch starts at zero, so training begins exactly at the pretrained decoder
        # and depth is used only as far as it helps.
        self.dpt_blks = None
        if depth:
            self.dpt_blks = nn.ModuleList(
                ImageProjection(in_channels // 3, out_channels, inter_channels=16)
                for in_channels, out_channels in zip(
                    ipt_blk_in_channels,
                    [ipt_blk_out_channels[0], *ipt_blk_out_channels],
                )
            )
            for block in self.dpt_blks:
                nn.init.zeros_(block.conv_out.weight)
                nn.init.zeros_(block.conv_out.bias)

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
        # All class interfaces contribute; background class 0 is not privileged.
        probs = logits.detach().softmax(dim=1)
        dx = F.pad((probs[..., 1:] - probs[..., :-1]).abs().sum(1, keepdim=True), (0, 1))
        dy = F.pad((probs[..., 1:, :] - probs[..., :-1, :]).abs().sum(1, keepdim=True), (0, 0, 0, 1))
        edge = F.interpolate(dx + dy, size=gradient.shape[2:], mode="bilinear", align_corners=True)
        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gradient = gradient.detach().abs()
        gradient = gradient / gradient.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (gradient * edge).detach()

    def _project(
        self, level: int, x: torch.Tensor, depth: torch.Tensor | None, feature: torch.Tensor
    ) -> torch.Tensor:
        """Image patches (plus depth, when built with it) concatenated onto a feature map."""
        blocks = (self.ipt_blk5, self.ipt_blk4, self.ipt_blk3, self.ipt_blk2, self.ipt_blk1)
        patches = blocks[level](x, feature)
        if self.dpt_blks is not None:
            if depth is None:
                raise ValueError("This decoder was built with depth; pass a depth map")
            patches = patches + self.dpt_blks[level](depth, feature)
        return torch.cat((feature, patches), dim=1)

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        depth: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        """Eval returns the full-size logits; training also returns the side logits
        before them and the gradient guidance (predictions, labels) per stage."""
        if self.training:
            x, x1, x2, x3, x4, gradient = features
        else:
            x, x1, x2, x3, x4 = features
        stages = (
            (self.decoder_block4, self.conv_ms_spvn_4, self.gdt_convs_4,
             self.gdt_convs_pred_4, self.gdt_convs_attn_4, self.lateral_block4, x3),
            (self.decoder_block3, self.conv_ms_spvn_3, self.gdt_convs_3,
             self.gdt_convs_pred_3, self.gdt_convs_attn_3, self.lateral_block3, x2),
            (self.decoder_block2, self.conv_ms_spvn_2, self.gdt_convs_2,
             self.gdt_convs_pred_2, self.gdt_convs_attn_2, self.lateral_block2, x1),
        )
        side, gdt_preds, gdt_labels = [], [], []
        p = self._project(0, x, depth, x4)
        for level, (block, head, gdt, gdt_pred, gdt_attn, lateral, skip) in enumerate(stages, 1):
            p = run_block(block, p, self.grad_checkpoint)
            p_gdt = gdt(p)
            if self.training:
                logits = head(p)
                side.append(logits)
                gdt_labels.append(self.guidance(logits, gradient))
                gdt_preds.append(gdt_pred(p_gdt))
            p = p * gdt_attn(p_gdt).sigmoid()
            p = F.interpolate(p, size=skip.shape[2:], mode="bilinear", align_corners=True)
            p = self._project(level, x, depth, p + lateral(skip))

        p = run_block(self.decoder_block1, p, self.grad_checkpoint)
        p = F.interpolate(p, size=x.shape[2:], mode="bilinear", align_corners=True)
        logits = [*side, self.conv_out1(self._project(4, x, depth, p))]
        if not self.training:
            return logits
        return [gdt_preds, gdt_labels], logits
