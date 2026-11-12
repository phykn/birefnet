import random

import cv2
import numpy as np


def pick_painted_pixel(label):
    painted = np.unique(label[label != 255])
    if painted.size == 0:
        return None
    ys, xs = np.nonzero(label == random.choice(painted.tolist()))
    pick = random.randrange(len(ys))
    return int(ys[pick]), int(xs[pick])


def pick_crop(
    shape,
    limits,
    crop=1024,
    focus=None,
):
    height, width = shape
    low, high = limits
    scale = float(np.exp(random.uniform(np.log(low), np.log(high))))
    side = max(1, round(max(height, width) / scale * crop / 1024))
    crop_h = min(side, height)
    crop_w = min(side, width)
    if focus is None:
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
    else:
        y, x = focus
        top = random.randint(max(0, y - crop_h + 1), min(y, height - crop_h))
        left = random.randint(max(0, x - crop_w + 1), min(x, width - crop_w))
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
    return np.s_[top:bottom, left:right], cut, side


def apply_gamma(image, limits):
    low, high = limits
    value = float(np.exp(random.uniform(np.log(low), np.log(high))))
    x = (image.astype(np.float32) / 255.0) ** value
    return np.rint(x * 255.0).astype(np.uint8)


def apply_shading(image, strength):
    height, width = image.shape[:2]
    angle = random.uniform(0.0, 2.0 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = np.cos(angle) * xs + np.sin(angle) * ys
    ramp -= ramp.min()
    ramp = ramp / ramp.max() * 2.0 - 1.0 if ramp.max() > 0 else ramp
    gain = 1.0 + random.uniform(-strength, strength) * ramp
    x = image.astype(np.float32) * gain
    return np.rint(np.clip(x, 0.0, 255.0)).astype(np.uint8)


def apply_blur(image, strength):
    sigma = random.uniform(0.0, strength)
    if sigma < 0.1:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def apply_noise(image, strength):
    std = random.uniform(0.0, strength) * 255.0
    rng = np.random.default_rng(random.getrandbits(32))
    x = image.astype(np.float32) + rng.normal(0.0, std, image.shape).astype(np.float32)
    return np.rint(np.clip(x, 0.0, 255.0)).astype(np.uint8)


def hide_edges(image, mask, *others):
    roi = mask != 255
    count = np.count_nonzero(roi)
    height, width = mask.shape
    for _ in range(32):
        top, bottom = (random.randint(0, height // 4) for _ in range(2))
        left, right = (random.randint(0, width // 4) for _ in range(2))
        region = np.s_[top : height - bottom, left : width - right]
        if np.count_nonzero(roi[region]) * 2 < count:
            continue
        hidden = np.ones(mask.shape, dtype=bool)
        hidden[region] = False
        value = random.choice((0, 255, 127, random.randint(0, 255)))
        image, mask = image.copy(), mask.copy()
        image[hidden] = value
        mask[hidden] = 255
        others = tuple(np.where(hidden, 255, other) for other in others)
        return image, mask, *others
    return image, mask, *others


def rotate_flip(*items):
    turns = int(np.random.randint(4))
    mirror = bool(np.random.randint(2))
    result = []
    for item in items:
        item = np.rot90(item, k=turns)
        if mirror:
            item = np.fliplr(item)
        result.append(np.ascontiguousarray(item))
    return tuple(result)
