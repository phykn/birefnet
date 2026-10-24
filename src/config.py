from pathlib import Path

from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Config must be a mapping: {path}")
    if "data" in cfg and "sem" in cfg.data and not isinstance(cfg.data.sem, bool):
        raise ValueError("data.sem must be a bool")
    return cfg


def load_config(path: str | Path | None = None) -> DictConfig:
    cfg = read_config(ROOT / "config.yaml")
    if path is not None:
        cfg = OmegaConf.merge(cfg, read_config(path))
    return cfg
