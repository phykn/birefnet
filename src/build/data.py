import os
from glob import glob
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data.dataset import MaskDataset
from ..data.pairs import pair_files
from ..data.split import Splits, make, restore
from ..prepare.spec import PreprocessSpec


def _loader(cfg: Any, dataset: MaskDataset) -> DataLoader:
    workers = cfg.loader.num_workers
    if type(workers) is not int or workers < 0:
        raise ValueError("loader.num_workers must be a non-negative integer")
    for name, default in (("batch_size", None), ("prefetch_factor", 2)):
        value = cfg.loader.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"loader.{name} must be a positive integer")
    for name in ("persistent_workers", "pin_memory", "drop_last"):
        if not isinstance(cfg.loader.get(name, True), bool):
            raise ValueError(f"loader.{name} must be a bool")
    batch = cfg.loader.batch_size

    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch,
        "shuffle": True,
        "drop_last": cfg.loader.get("drop_last", True),
        "num_workers": workers,
        # Pinned memory only speeds up copies to CUDA; elsewhere it is unsupported.
        "pin_memory": cfg.loader.get("pin_memory", True) and torch.cuda.is_available(),
    }
    if options["drop_last"] and len(dataset) < batch:
        raise ValueError("loader.drop_last would leave no training batch; lower loader.batch_size")
    # Worker options mean nothing when the main process loads the data.
    if workers > 0:
        options.update(
            persistent_workers=cfg.loader.get("persistent_workers", True),
            prefetch_factor=cfg.loader.get("prefetch_factor", 2),
        )
    return DataLoader(**options)


def build(cfg: Any, splits: Splits | None = None) -> tuple[DataLoader, Splits]:
    """The training loader and the train/valid split; validation reads its pairs directly."""
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
    dataset = MaskDataset(
        pairs=groups["train"],
        num_classes=int(cfg.model.classes),
        ignore_index=int(cfg.data.get("ignore_index", 255)),
        size=PreprocessSpec.size,
        is_sem=cfg.data.get("sem", False),
        zoom=tuple(cfg.augment.get("zoom", (1.0, 1.0))),
        brightness=cfg.augment.brightness,
        contrast=cfg.augment.contrast,
        gamma=tuple(cfg.augment.get("gamma", (1.0, 1.0))),
        illumination=float(cfg.augment.get("light", 0.0)),
        blur=float(cfg.augment.get("blur", 0.0)),
        noise=float(cfg.augment.get("noise", 0.0)),
        masking_prob=float(cfg.augment.get("hide_edges", 0.0)),
    )
    return _loader(cfg, dataset), groups
