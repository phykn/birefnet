from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Depth Anything's working resolution: 37 patches of 14 pixels.
DEPTH_SIZE = 518


class Depth(nn.Module):
    """Frozen relative-depth estimator run on the model input itself.

    Depth is computed from the exact tensor the segmenter sees, after every
    augmentation, so image and depth always agree on geometry and brightness.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.net.requires_grad_(False)
        self.net.eval()

    @classmethod
    def load(cls, path: str | Path) -> "Depth":
        from transformers import AutoModelForDepthEstimation

        return cls(AutoModelForDepthEstimation.from_pretrained(str(path), local_files_only=True))

    def train(self, mode: bool = True) -> "Depth":
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Padding is exactly 0 after normalization. Image pixels never are, because
        # the ImageNet mean is not a multiple of 1/255.
        valid = (x != 0).any(dim=1, keepdim=True)
        # Both models expect ImageNet-normalized input, so x is passed as it is.
        small = F.interpolate(x, size=(DEPTH_SIZE, DEPTH_SIZE), mode="bicubic", align_corners=False)
        depth = self.net(pixel_values=small).predicted_depth[:, None].float()
        depth = F.interpolate(depth, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return rescale(depth, valid)


def rescale(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Map each image's depth to 0-1 over its own image pixels; padding stays 0.

    Relative depth has no fixed unit, so every image is scaled on its own, never
    across the batch. The 2nd/98th percentiles keep a few extreme pixels from
    flattening the rest.
    """
    out = torch.zeros_like(depth)
    bounds = torch.tensor([0.02, 0.98], device=depth.device)
    for index, (image, inside) in enumerate(zip(depth, valid)):
        values = image[inside]
        if values.numel() == 0:
            continue
        low, high = torch.quantile(values, bounds)
        scaled = (image - low) / (high - low).clamp_min(1e-6)
        out[index] = scaled.clamp(0.0, 1.0) * inside
    return out
