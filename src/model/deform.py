import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import deform_conv2d


def grid_deform_conv2d(input, offset, weight, padding, mask):
    batch, channels, height, width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h, out_w = offset.shape[-2:]
    taps = kernel_h * kernel_w
    offset = offset.view(batch, taps, 2, out_h, out_w)
    options = {"device": input.device, "dtype": offset.dtype}
    rows = torch.arange(out_h, **options) - padding
    cols = torch.arange(out_w, **options) - padding
    tap_y, tap_x = torch.meshgrid(
        torch.arange(kernel_h, **options),
        torch.arange(kernel_w, **options),
        indexing="ij",
    )
    y = rows.view(1, 1, -1, 1) + tap_y.reshape(1, taps, 1, 1) + offset[:, :, 0]
    x = cols.view(1, 1, 1, -1) + tap_x.reshape(1, taps, 1, 1) + offset[:, :, 1]
    grid = torch.stack(((2 * x + 1) / width - 1, (2 * y + 1) / height - 1), dim=-1)
    weight = weight.view(out_channels, channels, taps)
    out = 0
    for start in range(0, taps, 7):
        chunk = slice(start, start + 7)
        args = (input, grid[:, chunk], mask[:, chunk], weight[:, :, chunk])
        if torch.is_grad_enabled():
            out = out + checkpoint(sample_taps, *args, use_reentrant=False)
        else:
            out = out + sample_taps(*args)
    return out


def sample_taps(input, grid, mask, weight):
    batch, taps, out_h, out_w = grid.shape[:4]
    sampled = F.grid_sample(
        input,
        grid.reshape(batch, taps * out_h, out_w, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).view(batch, input.shape[1], taps, out_h, out_w)
    return torch.einsum("bckhw,ock->bohw", sampled * mask[:, None], weight)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        taps = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * taps, kernel_size, padding=self.padding
        )
        self.modulator_conv = nn.Conv2d(
            in_channels, taps, kernel_size, padding=self.padding
        )
        for conv in (self.offset_conv, self.modulator_conv):
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.regular_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=self.padding, bias=False
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = 2.0 * torch.sigmoid(self.modulator_conv(x))
        deform = grid_deform_conv2d if x.device.type == "mps" else deform_conv2d
        weight = self.regular_conv.weight
        return deform(x, offset, weight, padding=self.padding, mask=mask)
