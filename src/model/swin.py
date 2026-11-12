import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_windows(x, size):
    b, h, w, c = x.shape
    x = x.view(b, h // size, size, w // size, size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, size * size, c)


def merge_windows(windows, size, height, width):
    c = windows.shape[-1]
    x = windows.view(-1, height // size, width // size, size, size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, c)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        span = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(span * span, num_heads)
        )
        grid = torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        )
        coords = torch.stack(grid).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :] + window_size - 1
        self.register_buffer(
            "relative_position_index", relative[0] * span + relative[1]
        )
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        table = self.relative_position_bias_table
        bias = table[self.relative_position_index.view(-1)].view(n, n, -1)
        bias = bias.permute(2, 0, 1).to(q.dtype)
        if mask is not None:
            bias = bias + mask.to(q.dtype).unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x, mask, height, width):
        b, _, c = x.shape
        size, shift = self.window_size, self.shift_size
        shortcut = x
        x = self.norm1(x).view(b, height, width, c)
        x = F.pad(x, (0, 0, 0, -width % size, 0, -height % size))
        padded_h, padded_w = x.shape[1:3]
        if shift:
            x = torch.roll(x, (-shift, -shift), dims=(1, 2))
        windows = self.attn(
            split_windows(x, size), mask.repeat(b, 1, 1) if shift else None
        )
        x = merge_windows(windows, size, padded_h, padded_w)
        if shift:
            x = torch.roll(x, (shift, shift), dims=(1, 2))
        x = shortcut + x[:, :height, :width].reshape(b, height * width, c)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, height, width):
        b, _, c = x.shape
        x = F.pad(x.view(b, height, width, c), (0, 0, 0, width % 2, 0, height % 2))
        x = torch.cat(
            (x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]),
            dim=-1,
        )
        return self.reduction(self.norm(x.view(b, -1, 4 * c)))


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, downsample):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(dim, num_heads, window_size, (i % 2) * self.shift_size)
            for i in range(depth)
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, height, width):
        size, shift = self.window_size, self.shift_size
        regions = torch.zeros(1, -height // size * -size, -width // size * -size, 1)
        bands = (slice(0, -size), slice(-size, -shift), slice(-shift, None))
        for index, (rows, cols) in enumerate(itertools.product(bands, bands)):
            regions[:, rows, cols] = index
        windows = split_windows(regions.to(x.device), size).squeeze(-1)
        diff = windows[:, None, :] - windows[:, :, None]
        mask = torch.zeros_like(diff, dtype=x.dtype).masked_fill(
            diff != 0, float("-inf")
        )
        for block in self.blocks:
            x = block(x, mask, height, width)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.proj(F.pad(x, (0, -w % self.patch_size, 0, -h % self.patch_size)))
        height, width = x.shape[2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class SwinTransformer(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, window_size):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        dims = [embed_dim * 2**i for i in range(len(depths))]
        self.layers = nn.ModuleList(
            BasicLayer(dim, depth, heads, window_size, i < len(depths) - 1)
            for i, (dim, depth, heads) in enumerate(zip(dims, depths, num_heads))
        )
        for i, dim in enumerate(dims):
            self.add_module(f"norm{i}", nn.LayerNorm(dim))

    def forward(self, x):
        x, height, width = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, height, width)
            out = getattr(self, f"norm{i}")(x)
            outs.append(out.view(-1, height, width, out.shape[-1]).permute(0, 3, 1, 2))
            if layer.downsample is not None:
                x = layer.downsample(x, height, width)
                height, width = (height + 1) // 2, (width + 1) // 2
        return tuple(outs)


def build_large():
    return SwinTransformer(
        embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), window_size=12
    )
