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


def zoom(
    image: np.ndarray,
    mask: np.ndarray,
    limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pick a square window as if the image were taken at another magnification.

    Returns the window, its cut edges, and the window side in source pixels; the
    caller resizes that side to the model input, so zooming out leaves padding.
    """
    height, width = image.shape[:2]
    low, high = limits
    scale = float(np.exp(random.uniform(np.log(low), np.log(high))))
    side = max(1, round(max(height, width) / scale))
    crop_h = min(side, height)
    crop_w = min(side, width)
    top = random.randint(0, height - crop_h)
    left = random.randint(0, width - crop_w)
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
        side,
    )


def jitter(image: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    bright_max, contrast_max = limits
    bright = random.uniform(-bright_max, bright_max)
    contrast = random.uniform(-contrast_max, contrast_max)
    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) * (1.0 + contrast) + 0.5 + bright
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def gamma(image: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    value = float(np.exp(random.uniform(np.log(low), np.log(high))))
    x = (image.astype(np.float32) / 255.0) ** value
    return np.rint(x * 255.0).astype(np.uint8)


def illumination(image: np.ndarray, strength: float) -> np.ndarray:
    """Brighten one side and darken the other, like an uneven SEM detector signal."""
    height, width = image.shape[:2]
    angle = random.uniform(0.0, 2.0 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = np.cos(angle) * xs + np.sin(angle) * ys
    ramp -= ramp.min()
    ramp = ramp / ramp.max() * 2.0 - 1.0 if ramp.max() > 0 else ramp
    gain = 1.0 + random.uniform(-strength, strength) * ramp
    if image.ndim == 3:
        gain = gain[..., None]
    x = image.astype(np.float32) * gain
    return np.rint(np.clip(x, 0.0, 255.0)).astype(np.uint8)


def blur(image: np.ndarray, strength: float) -> np.ndarray:
    """Soften edges like a slightly unfocused SEM shot; strength is the largest sigma in pixels."""
    sigma = random.uniform(0.0, strength)
    if sigma < 0.1:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def noise(image: np.ndarray, strength: float) -> np.ndarray:
    """Add grain like a fast or low-current SEM scan; strength is the largest std as a 0-1 fraction."""
    std = random.uniform(0.0, strength) * 255.0
    # Drawn from Python's random state, which the data loader seeds per worker.
    rng = np.random.default_rng(random.getrandbits(32))
    x = image.astype(np.float32) + rng.normal(0.0, std, image.shape).astype(np.float32)
    return np.rint(np.clip(x, 0.0, 255.0)).astype(np.uint8)


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
