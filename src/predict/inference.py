import contextlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from ..prepare.convert import InputMode, convert
from ..prepare.fit import Fit, fit_tensor, restore
from .tile import plan, weigh

OutputMode = Literal["labels", "probability"]


def _autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )
    return torch.amp.autocast("cuda", dtype=dtype)


def _infer(
    model: torch.nn.Module,
    tensors: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with _autocast(device):
        logits = model(batch).logits[-1]
    if logits.ndim != 4 or logits.shape[1] != model.num_classes:
        raise RuntimeError("Model logits must have shape N,num_classes,H,W")
    return logits.float().cpu().numpy()


def _restore(logits: np.ndarray, fit: Fit) -> np.ndarray:
    return np.stack([restore(channel, fit) for channel in logits])


def _merge(
    model: torch.nn.Module,
    image: np.ndarray,
    grid: int,
    size: int,
    overlap: float,
    batch_size: int,
    device: torch.device,
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
            tensor, fit = fit_tensor(crop, size=size, mode="rgb")
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
    return merged / weight


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int = 1024,
    mode: InputMode = "rgb",
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
    processed = convert(image, mode=mode)
    total = None
    for grid in grids:
        if grid == 1:
            tensor, fit = fit_tensor(processed, size=size, mode="rgb")
            output = _restore(_infer(model, [tensor], device)[0], fit)
        else:
            output = _merge(
                model,
                processed,
                grid,
                size,
                overlap,
                tile_batch,
                device,
            )
        if total is None:
            total = output.astype(np.float32, copy=True)
        else:
            np.add(total, output, out=total)

    assert total is not None
    total /= np.float32(len(grids))
    return total


def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    output_mode: OutputMode = "labels",
    class_id: int | None = None,
    size: int = 1024,
    mode: InputMode = "rgb",
    tiles: Sequence[int] = (1,),
    overlap: float = 1 / 3,
    tile_batch: int = 2,
) -> np.ndarray:
    """Return class IDs, all float probabilities, or a selected uint8 probability."""
    if output_mode not in {"labels", "probability"}:
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    if class_id is not None:
        if output_mode != "probability":
            raise ValueError("class_id is only supported for probability output")
        if (not isinstance(class_id, int) or isinstance(class_id, bool)
                or not 0 <= class_id < model.num_classes):
            raise ValueError("class_id must be in [0, num_classes)")
    logits = predict_logits(
        model, image, size=size, mode=mode, tiles=tiles,
        overlap=overlap, tile_batch=tile_batch,
    )
    if output_mode == "labels":
        return logits.argmax(axis=0).astype(np.uint8)
    probs = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs /= probs.sum(axis=0, keepdims=True)
    if class_id is None:
        return probs
    return np.rint(probs[class_id] * 255.0).astype(np.uint8)
