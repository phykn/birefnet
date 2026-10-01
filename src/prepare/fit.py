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

    @property
    def region(self) -> tuple[slice, slice]:
        return (
            slice(self.top, self.top + self.dst_h),
            slice(self.left, self.left + self.dst_w),
        )

    def resize(self, image: np.ndarray, interp: int) -> np.ndarray:
        if (self.src_h, self.src_w) != image.shape[:2]:
            raise ValueError(
                "Fit source shape does not match image: "
                f"fit={(self.src_h, self.src_w)}, image={image.shape[:2]}"
            )
        return cv2.resize(image, (self.dst_w, self.dst_h), interpolation=interp)

    def pad(self, image: np.ndarray) -> np.ndarray:
        canvas = np.zeros((self.size, self.size, *image.shape[2:]), dtype=image.dtype)
        canvas[self.region] = image
        return canvas


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


def fit_tensor(
    image: np.ndarray,
    size: int = 1024,
    is_sem: bool = False,
    fit: Fit | None = None,
) -> tuple[np.ndarray, Fit]:
    fit = fit or plan(*image.shape[:2], size=size)
    x = convert(image, is_sem=is_sem)
    down = fit.dst_h < image.shape[0] or fit.dst_w < image.shape[1]
    interp = cv2.INTER_AREA if down else cv2.INTER_CUBIC
    x = normalize(fit.resize(x, interp))
    canvas = fit.pad(x)
    return np.transpose(canvas, (2, 0, 1)), fit


def fit_image(
    image: np.ndarray,
    size: int = 1024,
    is_sem: bool = False,
    fit: Fit | None = None,
) -> tuple[np.ndarray, np.ndarray, Fit]:
    tensor, fit = fit_tensor(image, size=size, is_sem=is_sem, fit=fit)
    valid = np.zeros((fit.size, fit.size), dtype=np.float32)
    valid[fit.region] = 1.0
    return tensor, valid[None], fit


def fit_mask(mask: np.ndarray, fit: Fit) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    resized = fit.resize(mask.astype(np.float32), cv2.INTER_NEAREST_EXACT)
    return fit.pad(resized)[None]


def restore(logit: np.ndarray, fit: Fit) -> np.ndarray:
    cropped = logit[fit.region]
    if cropped.shape == (fit.src_h, fit.src_w):
        return cropped.astype(np.float32, copy=False)
    return cv2.resize(
        cropped.astype(np.float32),
        (fit.src_w, fit.src_h),
        interpolation=cv2.INTER_LINEAR,
    )
