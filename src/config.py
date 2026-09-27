from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .prepare.spec import PreprocessSpec


ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Config must be a mapping: {path}")
    _migrate(cfg)
    _validate(cfg)
    return cfg


def _migrate(cfg: DictConfig) -> None:
    if "train" in cfg:
        if cfg.train.get("mode", "decoder") != "decoder":
            raise ValueError("Only decoder training is supported")
        for key in ("mode", "backbone_stages", "backbone_lr_scale", "freeze_bn"):
            cfg.train.pop(key, None)
    if "birefnet" in cfg:
        cfg.birefnet.pop("grad_checkpoint", None)
    cfg.pop("teacher", None)
    if "loss" in cfg and "region_loss" in cfg.loss:
        if cfg.loss.region_loss != "dice":
            raise ValueError("Region loss is fixed to Dice")
        del cfg.loss.region_loss
    if "data" in cfg:
        cfg.data.pop("boundary_prob", None)
        cfg.data.pop("boundary_crop_prob", None)
        if "size" in cfg.data:
            if type(cfg.data.size) is not int or cfg.data.size != PreprocessSpec.size:
                raise ValueError("Input size is fixed at 1024; remove data.size")
            del cfg.data.size
        for old, new in (("global_prob", "crop_prob"),
                         ("full_image_prob", "crop_prob")):
            if old in cfg.data:
                if new in cfg.data:
                    raise ValueError(f"Use data.{new} only; remove data.{old}")
                cfg.data[new] = 1.0 - float(cfg.data[old])
                del cfg.data[old]
        if "mode" in cfg.data:
            if "is_sem" in cfg.data:
                raise ValueError("Use data.is_sem only; remove data.mode")
            spec = PreprocessSpec.from_meta({"preprocess": {
                "size": PreprocessSpec.size, "mode": cfg.data.mode,
            }})
            cfg.data.is_sem = spec.is_sem
            del cfg.data.mode
    if "augment" in cfg and "edge_prob" in cfg.augment:
        if "masking_prob" in cfg.augment:
            raise ValueError("Use augment.masking_prob only; remove augment.edge_prob")
        cfg.augment.masking_prob = cfg.augment.edge_prob
        del cfg.augment.edge_prob
    if "augment" in cfg and "strong" in cfg.augment:
        if "brightness" in cfg.augment or "contrast" in cfg.augment:
            raise ValueError("Use augment.brightness and augment.contrast only")
        cfg.augment.brightness = cfg.augment.strong.brightness
        cfg.augment.contrast = cfg.augment.strong.contrast
        del cfg.augment.strong
        cfg.augment.pop("weak", None)


def _validate(cfg: DictConfig) -> None:
    if "data" not in cfg:
        return
    if "min_crop_size" in cfg.data:
        value = cfg.data.min_crop_size
        if type(value) is not int or not 1 <= value <= PreprocessSpec.size:
            raise ValueError("data.min_crop_size must be an integer in [1, 1024]")
    if "is_sem" in cfg.data and not isinstance(cfg.data.is_sem, bool):
        raise ValueError("data.is_sem must be a bool")


def load_config(path: str | Path | None = None) -> DictConfig:
    cfg = read_config(ROOT / "config/model.yaml")
    if path is not None:
        cfg = OmegaConf.merge(cfg, read_config(path))
    return cfg
