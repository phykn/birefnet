from dataclasses import asdict, dataclass, fields, is_dataclass
from math import isfinite
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Model:
    weight: str
    classes: int


@dataclass
class Data:
    image_dir: str
    mask_dir: str
    label_dir: str
    valid_ratio: float


@dataclass
class Gamma:
    p: float
    range: list


@dataclass
class Shading:
    p: float
    max: float


@dataclass
class Blur:
    p: float
    max_sigma: float


@dataclass
class Noise:
    p: float
    max_std: float


@dataclass
class HideEdges:
    p: float


@dataclass
class Augment:
    zoom: list
    gamma: Gamma
    shading: Shading
    blur: Blur
    noise: Noise
    hide_edges: HideEdges


@dataclass
class Loader:
    batch_size: int
    num_workers: int


@dataclass
class Train:
    steps: int
    val_every: int
    save_every: int
    lr: float
    head_lr_scale: float
    min_lr: float
    warmup: int
    weight_decay: float
    grad_clip: float
    accum: int
    ema: float
    crop: int


@dataclass
class Loss:
    ce: float
    gce_q: float
    dice: float
    boundary: float
    boundary_width: int
    aux: float
    label_blur: float
    teacher: float
    teacher_start: int
    label_weight: float
    mask_weight: float
    label_smooth: float


@dataclass
class Config:
    model: Model
    data: Data
    augment: Augment
    loader: Loader
    train: Train
    loss: Loss


def build_config(kind, values, prefix=""):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix or 'config'}: expected mapping")
    names = {f.name for f in fields(kind)}
    unknown = [key for key in values if key not in names]
    if unknown:
        path = f"{prefix}.{unknown[0]}" if prefix else unknown[0]
        raise ValueError(f"{path}: unknown key")
    parsed = {}
    for f in fields(kind):
        path = f"{prefix}.{f.name}" if prefix else f.name
        if f.name not in values:
            raise ValueError(f"{path}: missing key")
        value = values[f.name]
        if is_dataclass(f.type):
            parsed[f.name] = build_config(f.type, value, path)
            continue
        valid = type(value) is f.type
        if f.type is float:
            valid = type(value) in (int, float) and isfinite(value)
        if f.type is list:
            valid = (
                isinstance(value, list)
                and len(value) == 2
                and all(type(x) in (int, float) and isfinite(x) for x in value)
                and 0 < value[0] <= value[1]
            )
        if not valid:
            raise ValueError(f"{path}: invalid type or value {value!r}")
        parsed[f.name] = value
    return kind(**parsed)


def parse_config(values):
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    merge_config(base, values)
    cfg = build_config(Config, base)
    a, t = cfg.augment, cfg.train
    checks = {
        "model.classes": 2 <= cfg.model.classes <= 255,
        "data.valid_ratio": 0 < cfg.data.valid_ratio < 1,
        "loader.batch_size": cfg.loader.batch_size >= 1,
        "loader.num_workers": cfg.loader.num_workers >= 0,
        "train.steps": t.steps >= 1,
        "train.val_every": t.val_every >= 1,
        "train.save_every": t.save_every >= 1,
        "train.lr": t.lr > 0,
        "train.head_lr_scale": t.head_lr_scale >= 1,
        "train.min_lr": 0 <= t.min_lr <= t.lr,
        "train.warmup": 0 <= t.warmup < t.steps,
        "train.weight_decay": t.weight_decay >= 0,
        "train.grad_clip": t.grad_clip > 0,
        "train.accum": t.accum >= 1,
        "train.ema": 0 <= t.ema < 1,
        "train.crop": 64 <= t.crop <= 1024 and t.crop % 32 == 0,
        "loss.boundary": cfg.loss.boundary >= 0,
        "loss.boundary_width": cfg.loss.boundary_width >= 1,
        "loss.aux": cfg.loss.aux >= 0,
        "loss.label_blur": 0 <= cfg.loss.label_blur <= 10,
        "loss.teacher": 0 <= cfg.loss.teacher < 1,
        "loss.teacher_start": cfg.loss.teacher_start >= 1,
        "loss.ce": cfg.loss.ce >= 0,
        "loss.gce_q": 0 <= cfg.loss.gce_q <= 1,
        "loss.dice": cfg.loss.dice >= 0 and cfg.loss.ce + cfg.loss.dice > 0,
        "loss.label_weight": cfg.loss.label_weight >= 0,
        "loss.mask_weight": cfg.loss.mask_weight >= 0
        and cfg.loss.label_weight + cfg.loss.mask_weight > 0,
        "loss.label_smooth": 0 <= cfg.loss.label_smooth < 0.5,
        "augment.gamma.p": 0 <= a.gamma.p <= 1,
        "augment.shading.p": 0 <= a.shading.p <= 1,
        "augment.shading.max": 0 <= a.shading.max < 1,
        "augment.blur.p": 0 <= a.blur.p <= 1,
        "augment.blur.max_sigma": a.blur.max_sigma >= 0,
        "augment.noise.p": 0 <= a.noise.p <= 1,
        "augment.noise.max_std": 0 <= a.noise.max_std < 1,
        "augment.hide_edges.p": 0 <= a.hide_edges.p <= 1,
    }
    for key, valid in checks.items():
        if not valid:
            raise ValueError(f"{key}: outside allowed range")
    return cfg


def load_config(path=None):
    if path is None:
        return parse_config({})
    return parse_config(yaml.safe_load(resolve(path).read_text()))


def merge_config(base, override):
    if not isinstance(override, dict):
        raise ValueError("config: expected mapping")
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
