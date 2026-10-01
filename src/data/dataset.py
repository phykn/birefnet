from pathlib import Path

import numpy as np

from ..prepare.convert import convert, to_gray
from ..prepare.fit import fit_image, fit_mask
from .augment import crop, edge_mask, flip, jitter
from .image import read_image, read_mask
from .pairs import Pair


class MaskDataset:
    def __init__(
        self,
        pairs: list[Pair],
        size: int = 1024,
        train: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.4,
        is_sem: bool = False,
        crop_prob: float = 0.7,
        num_classes: int = 4,
        ignore_index: int = 255,
        masking_prob: float = 0.0,
        min_crop_size: int = 256,
    ) -> None:
        if not 0.0 <= crop_prob <= 1.0:
            raise ValueError("crop_prob must be in [0, 1]")
        if not 0.0 <= masking_prob <= 1.0:
            raise ValueError("masking_prob must be in [0, 1]")
        self.pairs = pairs
        self.size = int(size)
        self.train = train
        if train and (type(min_crop_size) is not int or not 1 <= min_crop_size <= self.size):
            raise ValueError("min_crop_size must be an integer in [1, size]")
        self.min_crop_size = min_crop_size
        self.jitter = (brightness, contrast)
        if not isinstance(is_sem, bool):
            raise ValueError("is_sem must be a bool")
        self.is_sem = is_sem
        self.crop_prob = float(crop_prob)
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

        if self.train:
            image, mask, cut = crop(
                image,
                mask,
                self.size,
                self.crop_prob,
                self.min_crop_size,
            )
            image, mask, cut = flip(image, mask, cut)
            image = jitter(image, self.jitter)
        else:
            cut = np.zeros(mask.shape[:2], dtype=np.uint8)
        image = convert(image, is_sem=self.is_sem)
        if self.train and np.random.random() < self.masking_prob:
            image, mask = edge_mask(image, mask, self.ignore_index)

        image, valid, fit = fit_image(
            image,
            size=self.size,
            is_sem=False,
        )
        labels = fit_mask(mask, fit)[0].astype(np.int64)
        labels[valid[0] == 0] = self.ignore_index
        valid = valid * (labels[None] != self.ignore_index)
        sample = {
            "image": image,
            "mask": labels,
            "valid": valid,
            "cut": fit_mask(cut, fit),
        }
        return sample

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
