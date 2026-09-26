import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import laplacian

from .decoder.block import BasicDecBlk
from .decoder.net import Decoder
from .swin import build_large
from .output import Output


class BiRefNet(nn.Module):
    def __init__(
        self,
        channels: list[int] = [1536, 768, 384, 192],
        grad_checkpoint: bool = False,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        self.num_classes = num_classes
        self.freeze_bn = True
        self.finetune_mode = "decoder"
        self.backbone_stages = 1

        channels = [channel * 2 for channel in channels]

        self.cxt = channels[1:][::-1][-3:]

        self.bb = build_large(grad_checkpoint=grad_checkpoint)

        self.squeeze_module = nn.Sequential(
            BasicDecBlk(channels[0] + sum(self.cxt), channels[0])
        )

        self.decoder = Decoder(
            channels=channels,
            num_classes=num_classes,
        )
        self.configure_finetune()

    def configure_finetune(
        self, mode: str = "decoder", backbone_stages: int = 1, freeze_bn: bool = True
    ) -> None:
        if mode not in {"decoder", "partial", "full"}:
            raise ValueError("train.mode must be decoder, partial, or full")
        if mode == "partial" and (
            isinstance(backbone_stages, bool) or not isinstance(backbone_stages, int)
            or not 1 <= backbone_stages <= len(self.bb.layers)
        ):
            raise ValueError("backbone_stages must select 1 to 4 final backbone stages")
        self.finetune_mode = mode
        self.backbone_stages = backbone_stages
        self.freeze_bn = freeze_bn
        for param in self.parameters():
            param.requires_grad_(mode == "full")
        for module in (self.squeeze_module, self.decoder):
            for param in module.parameters():
                param.requires_grad_(True)
        if mode == "partial":
            for idx in range(len(self.bb.layers) - backbone_stages, len(self.bb.layers)):
                for module in (self.bb.layers[idx], getattr(self.bb, f"norm{idx}")):
                    for param in module.parameters():
                        param.requires_grad_(True)
        self.train(self.training)

    @property
    def stats(self) -> dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def list_trainable(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def train(self, mode: bool = True) -> "BiRefNet":
        super().train(mode)
        if mode and self.finetune_mode != "full":
            self.bb.eval()
            if self.finetune_mode == "partial":
                for idx in range(len(self.bb.layers) - self.backbone_stages, len(self.bb.layers)):
                    self.bb.layers[idx].train()
                    getattr(self.bb, f"norm{idx}").train()
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.bb(x)
        h, w = x.shape[-2:]
        x_pyramid = F.interpolate(
            x, size=(h // 2, w // 2), mode="bilinear", align_corners=True
        )
        pyramid = self.bb(x_pyramid)
        x1, x2, x3, x4 = [
            torch.cat((feature, F.interpolate(
                small, size=feature.shape[2:], mode="bilinear", align_corners=True
            )), dim=1)
            for feature, small in zip(features, pyramid)
        ]

        if self.cxt:
            context = [
                F.interpolate(
                    feature,
                    size=x4.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                for feature in (x1, x2, x3)[-len(self.cxt):]
            ]
            x4 = torch.cat((*context, x4), dim=1)

        return (x1, x2, x3, x4)

    def forward(self, x: torch.Tensor) -> Output:
        x1, x2, x3, x4 = self.encode(x)
        x4 = self.squeeze_module(x4)
        features = [x, x1, x2, x3, x4]
        if self.training:
            features.append(laplacian(x.mean(dim=1, keepdim=True), kernel_size=5))
        result = self.decoder(features)
        if self.training:
            gdt, logits = result
            return Output(logits=logits, gdt=(gdt[0], gdt[1]))
        return Output(logits=result)
