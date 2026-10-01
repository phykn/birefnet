import os
from glob import glob
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data.dataset import MaskDataset
from ..data.pairs import pair_files
from ..data.split import Splits, make, restore
from ..prepare.spec import PreprocessSpec


def _loader(cfg: Any, dataset: MaskDataset, shuffle: bool) -> DataLoader:
    workers = int(cfg.loader.num_workers)
    if workers < 0:
        raise ValueError("loader.num_workers must be non-negative")

    persistent = bool(cfg.loader.get("persistent_workers", workers > 0))
    if workers == 0 and persistent:
        raise ValueError("persistent_workers requires num_workers > 0")

    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(cfg.loader.batch),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(cfg.loader.get("pin_memory", torch.cuda.is_available())),
    }
    if workers > 0:
        prefetch = int(cfg.loader.get("prefetch_factor", 2))
        if prefetch < 1:
            raise ValueError("loader.prefetch_factor must be positive")
        options.update(
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    return DataLoader(**options)


def build(
    cfg: Any,
    splits: Splits | None = None,
) -> tuple[DataLoader, DataLoader, Splits]:
    image_paths = glob(os.path.join(cfg.data.image_dir, "*"))
    mask_paths = glob(os.path.join(cfg.data.mask_dir, "*"))
    pairs = pair_files(image_paths, mask_paths)

    if len(pairs) < 2:
        raise ValueError(
            "At least two image/mask pairs are required for train/valid"
        )

    groups = (
        make(pairs, float(cfg.data.valid_ratio))
        if splits is None
        else restore(pairs, splits)
    )
    train_data = groups["train"]
    valid_data = groups["valid"]
    label_options = dict(num_classes=int(cfg.birefnet.num_classes),
                         ignore_index=int(cfg.data.get("ignore_index", 255)))

    train_dataset = MaskDataset(
        pairs=train_data,
        **label_options,
        size=PreprocessSpec.size,
        train=True,
        is_sem=cfg.data.get("is_sem", False),
        crop_prob=float(cfg.data.get("crop_prob", 0.7)),
        min_crop_size=cfg.data.get("min_crop_size", 256),
        brightness=cfg.augment.brightness,
        contrast=cfg.augment.contrast,
        masking_prob=float(cfg.augment.get("masking_prob", 0.0)),
    )
    valid_dataset = MaskDataset(
        pairs=valid_data,
        **label_options,
        size=PreprocessSpec.size,
        train=False,
        is_sem=cfg.data.get("is_sem", False),
    )

    train_loader = _loader(cfg, train_dataset, shuffle=True)
    valid_loader = _loader(cfg, valid_dataset, shuffle=False)
    return train_loader, valid_loader, groups
