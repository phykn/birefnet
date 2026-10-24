from collections.abc import Sequence

import numpy as np
import torch

from ..prepare.fit import Fit, fit_tensor, restore
from ..precision import autocast, autocast_dtype
from .tile import plan, weigh


def _infer(
    model: torch.nn.Module,
    tensors: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with autocast(device, dtype=autocast_dtype(device), enabled=device.type == "cuda"):
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
    is_sem: bool = False,
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
    is_sem: bool = False,
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
