import random

import cv2
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
    global_prob: float,
    boundary_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if random.random() < global_prob or (height <= size and width <= size):
        return image, mask, np.zeros(mask.shape[:2], dtype=np.uint8)

    crop_h = min(size, height)
    crop_w = min(size, width)
    if random.random() < boundary_prob:
        labels = mask.astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary = cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)
        ys, xs = np.nonzero(boundary)
    else:
        ys = xs = np.empty(0, dtype=np.int64)

    if len(ys):
        pick = random.randrange(len(ys))
        center_y, center_x = int(ys[pick]), int(xs[pick])
    else:
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
