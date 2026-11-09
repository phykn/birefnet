import math
from dataclasses import dataclass
import numpy as np
from collections.abc import Sequence
import torch
from .device import amp_enabled, autocast, autocast_dtype
from .prepare import Fit, fit_tensor, restore


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

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


def _plan_axis(
    length: int, count: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if not 1 / 3 <= overlap < 1.0:
        raise ValueError("overlap must be in [1/3, 1)")
    if count == 1:
        return [(0, length, 0, 0)]

    extent = math.ceil(length / (count - (count - 1) * overlap))
    while True:
        starts = np.rint(np.linspace(0, length - extent, count)).astype(int)
        min_overlap = math.ceil(extent * overlap)
        actual = extent - int(np.diff(starts).max())
        if actual >= min_overlap:
            break
        extent += 1

    planned = []
    for index, start in enumerate(starts):
        start = int(start)
        end = start + extent
        before = 0 if index == 0 else int(starts[index - 1]) + extent - start
        after = 0 if index == len(starts) - 1 else end - int(starts[index + 1])
        planned.append((start, end, before, after))
    return planned


def plan(
    height: int,
    width: int,
    grid: int,
    overlap: float = 1 / 3,
) -> list[Tile]:
    if grid <= 0:
        raise ValueError("grid must be positive")
    if not 1 / 3 <= overlap < 1.0:
        raise ValueError("overlap must be in [1/3, 1)")

    ys = _plan_axis(height, grid, overlap)
    xs = _plan_axis(width, grid, overlap)
    return [
        Tile(
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            overlap_top=overlap_top,
            overlap_left=overlap_left,
            overlap_bottom=overlap_bottom,
            overlap_right=overlap_right,
        )
        for top, bottom, overlap_top, overlap_bottom in ys
        for left, right, overlap_left, overlap_right in xs
    ]


def _axis_weight(length: int, before: int, after: int) -> np.ndarray:
    weight = np.ones(length, dtype=np.float32)
    for count, view in ((before, weight), (after, weight[::-1])):
        if count:
            angles = np.linspace(0.0, np.pi, count + 2, dtype=np.float32)[1:-1]
            view[:count] = np.minimum(view[:count], 0.5 - 0.5 * np.cos(angles))
    return weight


def weigh(tile: Tile) -> np.ndarray:
    wy = _axis_weight(tile.height, tile.overlap_top, tile.overlap_bottom)
    wx = _axis_weight(tile.width, tile.overlap_left, tile.overlap_right)
    return wy[:, None] * wx[None, :]


def _infer(
    model: torch.nn.Module,
    tensors: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with autocast(device, dtype=autocast_dtype(device), enabled=amp_enabled(device)):
        logits = model(batch).logits[-1]
    if logits.ndim != 4 or logits.shape[1] != model.num_classes:
        raise RuntimeError("Model logits must have shape N,num_classes,H,W")
    return logits.float().cpu().numpy()


def _restore(logits: np.ndarray, fit: Fit) -> np.ndarray:
    output = np.empty((len(logits), fit.src_h, fit.src_w), dtype=np.float32)
    for index, channel in enumerate(logits):
        output[index] = restore(channel, fit)
    return output


def _merge(
    model: torch.nn.Module,
    image: np.ndarray,
    grid: int,
    size: int,
    overlap: float,
    batch_size: int,
    device: torch.device,
    is_sem: bool = True,
) -> np.ndarray:
    height, width = image.shape[:2]
    boxes = plan(height, width, grid=grid, overlap=overlap)
    merged = np.zeros((model.num_classes, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start : start + batch_size]
        tensors = []
        fits = []
        for box in chunk:
            crop = image[box.top : box.bottom, box.left : box.right]
            tensor, fit = fit_tensor(crop, size=size, is_sem=is_sem)
            tensors.append(tensor)
            fits.append(fit)

        logits = _infer(model, tensors, device)
        for index, (box, fit) in enumerate(zip(chunk, fits)):
            blend = weigh(box)
            region = np.s_[box.top : box.bottom, box.left : box.right]
            merged[:, region[0], region[1]] += _restore(logits[index], fit) * blend
            weight[region] += blend

    if np.any(weight <= 0):
        raise RuntimeError("Tile planner left pixels without merge weight")
    merged /= weight
    return merged


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int = 1024,
    is_sem: bool = True,
    tiles: Sequence[int] = (1,),
    overlap: float = 1 / 3,
    tile_batch: int = 2,
) -> np.ndarray:
    if not isinstance(model.num_classes, int) or not 2 <= model.num_classes <= 256:
        raise ValueError("num_classes must be an integer in [2, 256]")
    grids = tuple(tiles)
    if not grids:
        raise ValueError("tiles must not be empty")
    if any(
        not isinstance(grid, int) or isinstance(grid, bool) or grid <= 0
        for grid in grids
    ):
        raise ValueError("tiles must contain positive integers")
    if tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    if not 1 / 3 <= overlap < 1.0:
        raise ValueError("overlap must be in [1/3, 1)")

    if model.training:
        model.eval()
    device = next(model.parameters()).device
    total = None
    for grid in grids:
        if grid == 1:
            tensor, fit = fit_tensor(image, size=size, is_sem=is_sem)
            output = _restore(_infer(model, [tensor], device)[0], fit)
        else:
            output = _merge(
                model,
                image,
                grid,
                size,
                overlap,
                tile_batch,
                device,
                is_sem,
            )
        if total is None:
            total = output
        else:
            np.add(total, output, out=total)

    assert total is not None
    total /= np.float32(len(grids))
    return total


def probabilities(logits: np.ndarray) -> np.ndarray:
    probs = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs /= probs.sum(axis=0, keepdims=True)
    return probs
