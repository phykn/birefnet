import random

import numpy as np


def flip(*items: np.ndarray) -> tuple[np.ndarray, ...]:
    turns = int(np.random.randint(4))
    mirror = bool(np.random.randint(2))
    return tuple(_flip(item, turns, mirror) for item in items)


def _flip(x: np.ndarray, turns: int, mirror: bool) -> np.ndarray:
    x = np.rot90(x, k=turns)
    if mirror:
        x = np.fliplr(x)
    return np.ascontiguousarray(x)


def crop(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    crop_prob: float,
    min_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if random.random() >= crop_prob:
        return image, mask, np.zeros(mask.shape[:2], dtype=np.uint8)

    side = random.randint(min_size, size)
    crop_h = min(side, height)
    crop_w = min(side, width)
    center_y = random.randrange(height)
    center_x = random.randrange(width)

    top = min(max(center_y - crop_h // 2, 0), height - crop_h)
    left = min(max(center_x - crop_w // 2, 0), width - crop_w)
    bottom, right = top + crop_h, left + crop_w
    cut = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if top > 0:
        cut[0] = 1
    if bottom < height:
        cut[-1] = 1
    if left > 0:
        cut[:, 0] = 1
    if right < width:
        cut[:, -1] = 1
    return (
        image[top:bottom, left:right].copy(),
        mask[top:bottom, left:right].copy(),
        cut,
    )


def jitter(image: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    bright_max, contrast_max = limits
    bright = random.uniform(-bright_max, bright_max)
    contrast = random.uniform(-contrast_max, contrast_max)
    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) * (1.0 + contrast) + 0.5 + bright
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def edge_mask(
    image: np.ndarray,
    mask: np.ndarray,
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    roi = mask != ignore_index
    count = np.count_nonzero(roi)
    if count == 0:
        return image, mask
    height, width = mask.shape
    # A bounded retry also handles sparse ROIs at the image border.
    for _ in range(32):
        top, bottom = (random.randint(0, height // 4) for _ in range(2))
        left, right = (random.randint(0, width // 4) for _ in range(2))
        region = np.s_[top:height - bottom, left:width - right]
        if np.count_nonzero(roi[region]) * 2 < count:
            continue
        hidden = np.ones(mask.shape, dtype=bool)
        hidden[region] = False
        value = random.choice((0, 255, 127, random.randint(0, 255)))
        image, mask = image.copy(), mask.copy()
        image[hidden] = value
        mask[hidden] = ignore_index
        return image, mask
    return image, mask
