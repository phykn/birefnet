"""Shared colors and contact sheets for local inspection scripts."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.prepare.convert import MEAN, STD


def palette(num_classes: int) -> np.ndarray:
    if not 2 <= num_classes <= 256:
        raise ValueError("Preview supports 2 to 256 classes")
    colors = np.random.default_rng(0).integers(32, 240, (256, 3), dtype=np.uint8)
    colors[:4] = [(24, 32, 48), (174, 187, 199), (239, 89, 79), (64, 184, 220)]
    return colors


def image_from_tensor(tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.rint(np.clip(image * STD + MEAN, 0, 1) * 255).astype(np.uint8)


def colorize(labels: np.ndarray, num_classes: int, valid=None) -> np.ndarray:
    active = np.ones(labels.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ((labels[active] < 0) | (labels[active] >= num_classes)).any():
        raise ValueError("Mask contains invalid class IDs")
    colors = np.full((*labels.shape, 3), 100, dtype=np.uint8)
    colors[active] = palette(num_classes)[labels[active]]
    return colors


def overlay(image: np.ndarray, colors: np.ndarray) -> np.ndarray:
    return np.rint(image.astype(np.float32) * 0.55 + colors * 0.45).astype(np.uint8)


def save_labels(labels: np.ndarray, num_classes: int, path: Path) -> None:
    image = Image.fromarray(labels.astype(np.uint8)).convert("P")
    image.putpalette(palette(num_classes).reshape(-1).tolist())
    image.save(path)


def save_preview(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    canvas = Image.new("RGB", (320 * len(panels), 352), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (title, pixels) in enumerate(panels):
        image = ImageOps.contain(Image.fromarray(pixels), (320, 320), Image.Resampling.NEAREST)
        left = index * 320
        draw.text((left + 10, 8), title, fill=(25, 25, 25))
        canvas.paste(image, (left + (320 - image.width) // 2, 32 + (320 - image.height) // 2))
    canvas.save(path)
