from dataclasses import dataclass

import torch
import torch.nn as nn
from kornia.filters import laplacian

from .decoder import BasicDecBlk, Decoder, resize_bilinear
from .swin import build_large


@dataclass
class Output:
    logits: list[torch.Tensor]
    gdt: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None


class BiRefNet(nn.Module):
    def __init__(self, num_classes, channels=(1536, 768, 384, 192), auxiliary=False):
        super().__init__()
        self.num_classes = num_classes
        self.auxiliary = auxiliary
        channels = [channel * 2 for channel in channels]
        self.bb = build_large()
        self.squeeze_module = nn.Sequential(
            BasicDecBlk(sum(channels), channels[0])
        )
        self.decoder = Decoder(channels=channels, num_classes=num_classes)
        self.bb.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.bb.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def encode(self, x):
        h, w = x.shape[-2:]
        small = self.bb(resize_bilinear(x, (h // 2, w // 2)))
        x1, x2, x3, x4 = [
            torch.cat((feature, resize_bilinear(half, feature.shape[2:])), dim=1)
            for feature, half in zip(self.bb(x), small)
        ]
        context = [resize_bilinear(feature, x4.shape[2:]) for feature in (x1, x2, x3)]
        x4 = torch.cat((*context, x4), dim=1)
        return x1, x2, x3, x4

    def forward(self, x, encoded=None):
        x1, x2, x3, x4 = encoded if encoded is not None else self.encode(x)
        features = (x, x1, x2, x3, self.squeeze_module(x4))
        gradient = None
        if self.training and self.auxiliary:
            gradient = laplacian(x.mean(dim=1, keepdim=True), kernel_size=5)
        logits, guidance = self.decoder(features, gradient)
        return Output(logits, guidance)
