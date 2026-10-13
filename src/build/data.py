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
    workers = cfg.loader.num_workers
    if type(workers) is not int or workers < 0:
        raise ValueError("loader.num_workers must be a non-negative integer")
    batch = cfg.loader.batch
    if type(batch) is not int or batch < 1:
        raise ValueError("loader.batch must be a positive integer")

    persistent = cfg.loader.get("persistent_workers", workers > 0)
    pin_memory = cfg.loader.get("pin_memory", torch.cuda.is_available())
    for name, value in (("persistent_workers", persistent), ("pin_memory", pin_memory)):
        if not isinstance(value, bool):
            raise ValueError(f"loader.{name} must be a bool")
    if workers == 0 and persistent:
        raise ValueError("persistent_workers requires num_workers > 0")

    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        prefetch = cfg.loader.get("prefetch_factor", 2)
        if type(prefetch) is not int or prefetch < 1:
            raise ValueError("loader.prefetch_factor must be a positive integer")
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
