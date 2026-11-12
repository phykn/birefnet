import cv2
import numpy as np
from dataclasses import dataclass


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def stretch_contrast(gray):
    x = gray.astype(np.float32)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    low, high = mean - 3.0 * std, mean + 3.0 * std
    x = (np.clip(x, low, high) - low) / (high - low)
    return np.rint(x * 255.0).astype(np.uint8)


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
    def region(self):
        return (
            slice(self.top, self.top + self.dst_h),
            slice(self.left, self.left + self.dst_w),
        )

    def resize(self, image, interp):
        return cv2.resize(image, (self.dst_w, self.dst_h), interpolation=interp)

    def pad(self, image, value=0):
        canvas = np.full(
            (self.size, self.size, *image.shape[2:]), value, dtype=image.dtype
        )
        canvas[self.region] = image
        return canvas


def plan_fit(height, width, size=1024, scale=None):
    if scale is None:
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


def fit_image(gray, size=1024, fit=None):
    fit = fit or plan_fit(*gray.shape, size=size)
    down = fit.dst_h < gray.shape[0] or fit.dst_w < gray.shape[1]
    interp = cv2.INTER_AREA if down else cv2.INTER_CUBIC
    resized = fit.resize(stretch_contrast(gray), interp)
    x = np.repeat(resized[..., None], 3, axis=2).astype(np.float32)
    x /= 255.0
    x -= MEAN
    x /= STD
    return np.transpose(fit.pad(x), (2, 0, 1)), fit


def fit_mask(mask, fit, padding=0):
    resized = fit.resize(mask.astype(np.float32), cv2.INTER_NEAREST_EXACT)
    return fit.pad(resized.astype(np.int64), padding)


def restore_logits(logits, fit):
    cropped = logits[:, fit.region[0], fit.region[1]].transpose(1, 2, 0)
    if cropped.shape[:2] != (fit.src_h, fit.src_w):
        cropped = cv2.resize(
            cropped, (fit.src_w, fit.src_h), interpolation=cv2.INTER_LINEAR
        )
    return np.ascontiguousarray(cropped.transpose(2, 0, 1), dtype=np.float32)
