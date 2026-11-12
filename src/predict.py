import math
from dataclasses import dataclass

import numpy as np
import torch

from .data.model_input import fit_image, restore_logits
from .device import autocast, pick_amp_dtype


@dataclass(frozen=True)
class Tile:
    top: int
    left: int
    bottom: int
    right: int
    overlap_top: int = 0
    overlap_left: int = 0
    overlap_bottom: int = 0
    overlap_right: int = 0


def plan_axis(length, count, overlap):
    if count == 1:
        return [(0, length, 0, 0)]
    extent = math.ceil(length / (count - (count - 1) * overlap))
    while True:
        starts = np.rint(np.linspace(0, length - extent, count)).astype(int)
        if extent - int(np.diff(starts).max()) >= math.ceil(extent * overlap):
            break
        extent += 1
    planned = []
    for index, start in enumerate(starts):
        start = int(start)
        before = 0 if index == 0 else int(starts[index - 1]) + extent - start
        after = 0 if index == count - 1 else start + extent - int(starts[index + 1])
        planned.append((start, start + extent, before, after))
    return planned


def plan_tiles(height, width, grid, overlap):
    return [
        Tile(top, left, bottom, right, before_y, before_x, after_y, after_x)
        for top, bottom, before_y, after_y in plan_axis(height, grid, overlap)
        for left, right, before_x, after_x in plan_axis(width, grid, overlap)
    ]


def weigh_axis(length, before, after):
    weight = np.ones(length, dtype=np.float32)
    for count, view in ((before, weight), (after, weight[::-1])):
        if count:
            angles = np.linspace(0.0, np.pi, count + 2, dtype=np.float32)[1:-1]
            view[:count] = np.minimum(view[:count], 0.5 - 0.5 * np.cos(angles))
    return weight


def weigh_tile(tile):
    wy = weigh_axis(tile.bottom - tile.top, tile.overlap_top, tile.overlap_bottom)
    wx = weigh_axis(tile.right - tile.left, tile.overlap_left, tile.overlap_right)
    return wy[:, None] * wx[None, :]


def infer(model, tensors, device):
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with autocast(device, pick_amp_dtype(device)):
        logits = model(batch).logits[-1]
    return logits.float().cpu().numpy()


def predict_tiles(model, image, grid, size, overlap, batch_size, device):
    height, width = image.shape
    tiles = plan_tiles(height, width, grid, overlap)
    merged = np.zeros((model.num_classes, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        fitted = [
            fit_image(image[t.top : t.bottom, t.left : t.right], size=size)
            for t in chunk
        ]
        logits = infer(model, [tensor for tensor, _ in fitted], device)
        for tile, (_, fit), logit in zip(chunk, fitted, logits):
            blend = weigh_tile(tile)
            region = np.s_[tile.top : tile.bottom, tile.left : tile.right]
            merged[:, region[0], region[1]] += restore_logits(logit, fit) * blend
            weight[region] += blend
    return merged / weight


@torch.inference_mode()
def predict(
    model,
    image,
    *,
    size=1024,
    tiles=(1,),
    overlap=1 / 3,
    tile_batch=2,
):
    if not tiles or min(tiles) <= 0:
        raise ValueError("tiles must contain positive integers")
    if tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    if not 1 / 3 <= overlap < 1.0:
        raise ValueError("overlap must be in [1/3, 1)")
    if model.training:
        model.eval()
    device = next(model.parameters()).device
    total = np.zeros((model.num_classes, *image.shape), dtype=np.float32)
    for grid in tiles:
        if grid == 1:
            tensor, fit = fit_image(image, size=size)
            total += restore_logits(infer(model, [tensor], device)[0], fit)
        else:
            total += predict_tiles(
                model, image, grid, size, overlap, tile_batch, device
            )
    logits = total / len(tiles)
    probs = np.exp(logits - logits.max(axis=0))
    probs /= probs.sum(axis=0)
    return {"logits": logits, "probs": probs, "classes": logits.argmax(axis=0)}
