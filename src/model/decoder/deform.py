import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torchvision.ops import deform_conv2d


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def grid_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """torchvision's modulated deform_conv2d built from grid_sample and einsum.

    torchvision has no MPS backward for deform_conv2d, so Apple GPUs would run it on
    the CPU. These ops all have MPS gradients. Covers one offset group, no dilation.
    """
    batch, channels, height, width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h, out_w = offset.shape[-2:]
    taps = kernel_h * kernel_w
    (stride_h, stride_w), (pad_h, pad_w) = _pair(stride), _pair(padding)
    if offset.shape[1] != 2 * taps:
        raise ValueError("Only one offset group is supported")

    # Sampling positions in input pixels: output position * stride - padding, plus
    # the kernel tap, plus the learned offset (dy, dx per tap, taps row-major).
    offset = offset.view(batch, taps, 2, out_h, out_w)
    rows = torch.arange(out_h, device=input.device, dtype=offset.dtype) * stride_h - pad_h
    cols = torch.arange(out_w, device=input.device, dtype=offset.dtype) * stride_w - pad_w
    tap_y, tap_x = torch.meshgrid(
        torch.arange(kernel_h, device=input.device, dtype=offset.dtype),
        torch.arange(kernel_w, device=input.device, dtype=offset.dtype),
        indexing="ij",
    )
    y = rows.view(1, 1, -1, 1) + tap_y.reshape(1, taps, 1, 1) + offset[:, :, 0]
    x = cols.view(1, 1, 1, -1) + tap_x.reshape(1, taps, 1, 1) + offset[:, :, 1]

    # With align_corners=False, pixel p sits at (2p + 1) / size - 1, and zero padding
    # matches torchvision's zero contribution from outside corners.
    grid = torch.stack(((2 * x + 1) / width - 1, (2 * y + 1) / height - 1), dim=-1)
    weight = weight.view(out_channels, channels, taps)
    # One tap at a time, recomputing its samples in backward: sampling every tap at
    # once holds taps x the input in memory, about 10 GB for a 7x7 kernel at 256 px.
    recompute = torch.is_grad_enabled()
    out = None
    for tap in range(taps):
        args = (input, grid[:, tap], None if mask is None else mask[:, tap : tap + 1], weight[:, :, tap])
        part = checkpoint(_tap, *args, use_reentrant=False) if recompute else _tap(*args)
        out = part if out is None else out + part
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _tap(
    input: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor | None, weight: torch.Tensor
) -> torch.Tensor:
    """One kernel tap: sample the input at its positions, then mix channels."""
    sampled = F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if mask is not None:
        sampled = sampled * mask
    return torch.einsum("bchw,oc->bohw", sampled, weight)


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        k_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * k_size[0] * k_size[1],
            kernel_size=k_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1 * k_size[0] * k_size[1],
            kernel_size=k_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        deform = grid_deform_conv2d if x.device.type == "mps" else deform_conv2d
        return deform(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
