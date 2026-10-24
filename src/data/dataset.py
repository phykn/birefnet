from math import isfinite
from pathlib import Path

import numpy as np

from ..prepare.convert import convert, to_gray
from ..prepare.fit import fit_image, fit_mask, plan
from .augment import blur, edge_mask, flip, gamma, illumination, jitter, noise, zoom
from .image import read_image, read_mask
from .pairs import Pair


class MaskDataset:
    """Training samples: each image/mask pair is augmented, converted and fitted to the input size."""

    def __init__(
        self,
        pairs: list[Pair],
        size: int = 1024,
        brightness: float = 0.2,
        contrast: float = 0.4,
        gamma: tuple[float, float] = (1.0, 1.0),
        illumination: float = 0.0,
        blur: float = 0.0,
        noise: float = 0.0,
        is_sem: bool = False,
        zoom: tuple[float, float] = (1.0, 1.0),
        num_classes: int = 4,
        ignore_index: int = 255,
        masking_prob: float = 0.0,
    ) -> None:
        if not 0.0 <= masking_prob <= 1.0:
            raise ValueError("masking_prob must be in [0, 1]")
        for name, value in (("brightness", brightness), ("contrast", contrast)):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, (low, high) in (("zoom", zoom), ("gamma", gamma)):
            if not (isfinite(low) and isfinite(high) and 0.0 < low <= high):
                raise ValueError(f"{name} must be a finite range with 0 < low <= high")
        if not 0.0 <= illumination < 1.0:
            raise ValueError("illumination must be in [0, 1)")
        if not (isfinite(blur) and blur >= 0.0):
            raise ValueError("blur must be finite and >= 0")
        if not 0.0 <= noise < 1.0:
            raise ValueError("noise must be in [0, 1)")
        self.pairs = pairs
        self.size = int(size)
        self.jitter = (brightness, contrast)
        self.zoom = (float(zoom[0]), float(zoom[1]))
        self.gamma = (float(gamma[0]), float(gamma[1]))
        self.illumination = float(illumination)
        self.blur = float(blur)
        self.noise = float(noise)
        if not isinstance(is_sem, bool):
            raise ValueError("is_sem must be a bool")
        self.is_sem = is_sem
        if num_classes < 2 or 0 <= ignore_index < num_classes:
            raise ValueError("num_classes must be >= 2 and ignore_index outside class indices")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.masking_prob = float(masking_prob)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image, mask = self._load(index)
        image = to_gray(image) if self.is_sem else convert(image)

        image, mask, cut, side = zoom(image, mask, self.zoom)
        image, mask, cut = flip(image, mask, cut)
        # Brightness, contrast, gamma, uneven light, focus and grain come before
        # the z-score, so training sees the same operator-dependent shifts as real
        # SEM shots.
        image = jitter(image, self.jitter)
        image = gamma(image, self.gamma)
        image = illumination(image, self.illumination)
        image = blur(image, self.blur)
        image = noise(image, self.noise)
        # Hidden edges stand in for an uncropped info bar or a dark rim, so they
        # also enter the z-score statistics, as such a bar would.
        if np.random.random() < self.masking_prob:
            image, mask = edge_mask(image, mask, self.ignore_index)
        image = convert(image, is_sem=self.is_sem)

        fit = plan(*image.shape[:2], size=self.size, scale=self.size / side)
        image, valid, fit = fit_image(image, size=self.size, fit=fit)
        labels = fit_mask(mask, fit)[0].astype(np.int64)
        labels[valid[0] == 0] = self.ignore_index
        valid = valid * (labels[None] != self.ignore_index)
        return {"image": image, "mask": labels, "valid": valid, "cut": fit_mask(cut, fit)}

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.pairs[index]
        image = read_image(image_path)
        mask = read_mask(mask_path)
        invalid = (mask != self.ignore_index) & ((mask < 0) | (mask >= self.num_classes))
        if invalid.any():
            raise ValueError(f"Mask has invalid class indices {np.unique(mask[invalid]).tolist()}: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Image and mask dimensions differ: "
                f"{Path(image_path).name}={image.shape[:2]}, "
                f"{Path(mask_path).name}={mask.shape[:2]}"
            )
        return image, mask
