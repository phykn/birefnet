from dataclasses import dataclass

import cv2
import numpy as np

from .convert import convert, normalize


@dataclass(frozen=True)
class Fit:
    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    top: int
    left: int
    size: int


def plan(height: int, width: int, size: int = 1024) -> Fit:
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError(
            f"height, width and size must be positive, got {(height, width, size)}"
        )
    scale = min(size / height, size / width)
    dst_h = min(size, max(1, int(round(height * scale))))
    dst_w = min(size, max(1, int(round(width * scale))))
    return Fit(
        src_h=height,
        src_w=width,
        dst_h=dst_h,
        dst_w=dst_w,
        top=(size - dst_h) // 2,
        left=(size - dst_w) // 2,
        size=size,
    )


def _resize(
    image: np.ndarray,
    fit: Fit,
    interp: int,
) -> np.ndarray:
    return cv2.resize(
        image,
        (fit.dst_w, fit.dst_h),
        interpolation=interp,
    )


def fit_tensor(
    image: np.ndarray,
    size: int = 1024,
    is_sem: bool = False,
    fit: Fit | None = None,
) -> tuple[np.ndarray, Fit]:
    fit = fit or plan(*image.shape[:2], size=size)
    if (fit.src_h, fit.src_w) != image.shape[:2]:
        raise ValueError(
            "Fit source shape does not match image: "
            f"fit={(fit.src_h, fit.src_w)}, image={image.shape[:2]}"
        )
    x = convert(image, is_sem=is_sem)
    down = fit.dst_h < image.shape[0] or fit.dst_w < image.shape[1]
    interp = cv2.INTER_AREA if down else cv2.INTER_CUBIC
    x = normalize(_resize(x, fit, interp))

    canvas = np.zeros((fit.size, fit.size, 3), dtype=np.float32)
    y0, x0 = fit.top, fit.left
    y1, x1 = y0 + fit.dst_h, x0 + fit.dst_w
    canvas[y0:y1, x0:x1] = x
    return np.transpose(canvas, (2, 0, 1)), fit


def fit_image(
    image: np.ndarray,
    size: int = 1024,
    is_sem: bool = False,
    fit: Fit | None = None,
) -> tuple[np.ndarray, np.ndarray, Fit]:
    tensor, fit = fit_tensor(image, size=size, is_sem=is_sem, fit=fit)
    valid = np.zeros((1, fit.size, fit.size), dtype=np.float32)
    y0, x0 = fit.top, fit.left
    y1, x1 = y0 + fit.dst_h, x0 + fit.dst_w
    valid[:, y0:y1, x0:x1] = 1.0
    return tensor, valid, fit


def fit_mask(mask: np.ndarray, fit: Fit) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    resized = _resize(mask.astype(np.float32), fit, cv2.INTER_NEAREST_EXACT)
    canvas = np.zeros((fit.size, fit.size), dtype=np.float32)
    y0, x0 = fit.top, fit.left
    y1, x1 = y0 + fit.dst_h, x0 + fit.dst_w
    canvas[y0:y1, x0:x1] = resized
    return canvas[None]


def restore(logit: np.ndarray, fit: Fit) -> np.ndarray:
    y0, x0 = fit.top, fit.left
    y1, x1 = y0 + fit.dst_h, x0 + fit.dst_w
    cropped = logit[y0:y1, x0:x1]
    if cropped.shape == (fit.src_h, fit.src_w):
        return cropped.astype(np.float32, copy=False)
    return cv2.resize(
        cropped.astype(np.float32),
        (fit.src_w, fit.src_h),
        interpolation=cv2.INTER_LINEAR,
    )
