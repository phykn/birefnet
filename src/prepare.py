import cv2
import numpy as np
from dataclasses import dataclass


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalize(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x /= 255.0
    x -= MEAN
    x /= STD
    return x


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got shape {image.shape}")
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[..., :3]
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)


def _zscore(gray: np.ndarray, std_range: float = 3.0) -> np.ndarray:
    x = gray.astype(np.float32)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    low, high = mean - std_range * std, mean + std_range * std
    x = np.clip(x, low, high)
    x = (x - low) / max(high - low, 1e-6)
    return np.rint(x * 255.0).astype(np.uint8)


def convert(image: np.ndarray, is_sem: bool = False) -> np.ndarray:
    if not isinstance(is_sem, bool):
        raise ValueError("is_sem must be a bool")
    rgb = _to_rgb(image)
    if not is_sem:
        return rgb
    # Keep the frozen backbone's three channels identical for SEM images.
    return np.repeat(_zscore(to_gray(rgb))[..., None], 3, axis=-1)


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

    def pad(self, image: np.ndarray, value=0) -> np.ndarray:
        canvas = np.full(
            (self.size, self.size, *image.shape[2:]), value, dtype=image.dtype
        )
        canvas[self.region] = image
        return canvas


def plan(height: int, width: int, size: int = 1024, scale: float | None = None) -> Fit:
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError(
            f"height, width and size must be positive, got {(height, width, size)}"
        )
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


def fit_mask(mask: np.ndarray, fit: Fit, padding=0) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    resized = fit.resize(mask.astype(np.float32), cv2.INTER_NEAREST_EXACT)
    return fit.pad(resized, padding)


def restore(logit: np.ndarray, fit: Fit) -> np.ndarray:
    cropped = logit[fit.region]
    if cropped.shape == (fit.src_h, fit.src_w):
        return cropped.astype(np.float32, copy=False)
    return cv2.resize(
        cropped.astype(np.float32),
        (fit.src_w, fit.src_h),
        interpolation=cv2.INTER_LINEAR,
    )
