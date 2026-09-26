from pathlib import Path

import numpy as np

from ..prepare.convert import InputMode, convert
from ..prepare.fit import fit_image, fit_mask, fit_tensor
from .augment import crop, flip, jitter
from .image import read_image, read_mask
from .pairs import Pair


class MaskDataset:
    def __init__(
        self,
        data: list[Pair],
        size: int = 1024,
        train: bool = False,
        weak: tuple[float, float] = (0.2, 0.2),
        strong: tuple[float, float] = (0.4, 0.4),
        mode: InputMode = "rgb",
        global_prob: float = 0.3,
        boundary_prob: float = 0.5,
        num_classes: int = 4,
        ignore_index: int = 255,
    ) -> None:
        if not 0.0 <= global_prob <= 1.0:
            raise ValueError("global_prob must be in [0, 1]")
        if not 0.0 <= boundary_prob <= 1.0:
            raise ValueError("boundary_prob must be in [0, 1]")
        self.data = data
        self.size = int(size)
        self.train = train
        self.weak = weak
        self.strong = strong
        self.mode = mode
        self.global_prob = float(global_prob)
        self.boundary_prob = float(boundary_prob)
        if num_classes < 2 or 0 <= ignore_index < num_classes:
            raise ValueError("num_classes must be >= 2 and ignore_index outside class indices")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

    @property
    def pairs(self) -> list[Pair]:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image, mask = self._load(index)
        image = convert(image, mode=self.mode)

        if self.train:
            image, mask, cut = crop(
                image,
                mask,
                self.size,
                self.global_prob,
                self.boundary_prob,
            )
            image, mask, cut = flip(image, mask, cut)
            weak = jitter(image, self.weak)
            strong = jitter(image, self.strong)
        else:
            cut = np.zeros(mask.shape[:2], dtype=np.uint8)
            weak = image

        weak, valid, fit = fit_image(
            weak,
            size=self.size,
            mode="rgb",
        )
        labels = fit_mask(mask, fit)[0].astype(np.int64)
        labels[valid[0] == 0] = self.ignore_index
        valid = valid * (labels[None] != self.ignore_index)
        sample = {
            "weak": weak,
            "mask": labels,
            "valid": valid,
            "cut": fit_mask(cut, fit),
        }
        if self.train:
            strong, strong_fit = fit_tensor(
                strong,
                size=self.size,
                mode="rgb",
                fit=fit,
            )
            if strong_fit != fit:
                raise RuntimeError(
                    "Two-view augmentation produced mismatched geometry."
                )
            sample["strong"] = strong
        return sample

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.data[index]
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
