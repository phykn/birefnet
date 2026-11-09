from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Model:
    weight: str = "weight/BiRefNet-general-epoch_244.pth"
    classes: int = 4
    depth: bool = False
    depth_weight: str = "weight/Depth-Anything-V2-Small-hf"


@dataclass
class Data:
    image_dir: str = "data/train/image"
    mask_dir: str = "data/train/mask"
    label_dir: str = ""
    valid_ratio: float = 0.1


@dataclass
class Augment:
    zoom: list = field(default_factory=lambda: [0.5, 2.0])
    hide_edges: float = 0.3
    brightness: float = 0.2
    contrast: float = 0.4
    gamma: list = field(default_factory=lambda: [0.7, 1.5])
    light: float = 0.2
    blur: float = 1.0
    noise: float = 0.05


@dataclass
class Loader:
    batch_size: int = 2
    num_workers: int = 4


@dataclass
class Train:
    seed: int = 0
    steps: int = 5000
    val_every: int = 250
    save_every: int = 500
    lr: float = 1.0e-4
    min_lr: float = 1.0e-6
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    accum: int = 4
    ema: float = 0.995
    crop: int = 512


@dataclass
class Loss:
    ce: float = 1.0
    dice: float = 1.0
    boundary: float = 0.0
    boundary_width: int = 3
    aux: float = 0.0
    label_blur: float = 0.0
    teacher: float = 0.0
    teacher_start: int = 1000
    label_weight: float = 1.0
    mask_weight: float = 3.0
    label_smooth: float = 0.05
    mask_edge: int = 4


@dataclass
class Config:
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    augment: Augment = field(default_factory=Augment)
    loader: Loader = field(default_factory=Loader)
    train: Train = field(default_factory=Train)
    loss: Loss = field(default_factory=Loss)


def parse(values, ignore_unknown=False):
    if not isinstance(values, dict):
        raise ValueError("config: expected mapping")
    cfg = Config()
    sections = {f.name for f in fields(cfg)}
    for section, content in values.items():
        if section not in sections:
            if ignore_unknown:
                continue
            raise ValueError(f"{section}: unknown key")
        if not isinstance(content, dict):
            raise ValueError(f"{section}: expected mapping")
        obj = getattr(cfg, section)
        types = {f.name: f.type for f in fields(obj)}
        for key, value in content.items():
            path = f"{section}.{key}"
            if key not in types:
                if ignore_unknown:
                    continue
                raise ValueError(f"{path}: unknown key")
            kind = types[key]
            valid = type(value) is kind
            if kind is float:
                valid = type(value) in (int, float) and isfinite(value)
            if kind is list:
                valid = (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(type(x) in (int, float) and isfinite(x) for x in value)
                    and 0 < value[0] <= value[1]
                )
            if not valid:
                raise ValueError(f"{path}: invalid type or value {value!r}")
            setattr(obj, key, value)
    a, t, l = cfg.augment, cfg.train, cfg.loader
    checks = {
        "model.classes": 2 <= cfg.model.classes <= 255,
        "data.valid_ratio": 0 < cfg.data.valid_ratio < 1,
        "loader.batch_size": l.batch_size >= 1,
        "loader.num_workers": l.num_workers >= 0,
        "train.seed": t.seed >= 0,
        "train.steps": t.steps >= 1,
        "train.val_every": t.val_every >= 1,
        "train.save_every": t.save_every >= 1,
        "train.lr": t.lr > 0,
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
        "loss.dice": cfg.loss.dice >= 0 and cfg.loss.ce + cfg.loss.dice > 0,
        "loss.label_weight": cfg.loss.label_weight >= 0,
        "loss.mask_weight": cfg.loss.mask_weight >= 0
        and cfg.loss.label_weight + cfg.loss.mask_weight > 0,
        "loss.label_smooth": 0 <= cfg.loss.label_smooth < 0.5,
        "loss.mask_edge": 0 <= cfg.loss.mask_edge <= 16,
        "augment.hide_edges": 0 <= a.hide_edges <= 1,
        "augment.brightness": 0 <= a.brightness <= 1,
        "augment.contrast": 0 <= a.contrast < 1,
        "augment.light": 0 <= a.light < 1,
        "augment.blur": a.blur >= 0,
        "augment.noise": 0 <= a.noise < 1,
    }
    for key, valid in checks.items():
        if not valid:
            raise ValueError(f"{key}: outside allowed range")
    return cfg


def read_config(path):
    return parse(yaml.safe_load(resolve(path).read_text()))


def load_config(path=None):
    values = yaml.safe_load((ROOT / "config.yaml").read_text())
    if path is not None:
        override = yaml.safe_load(resolve(path).read_text())
        if not isinstance(override, dict):
            raise ValueError("config: expected mapping")
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(values.get(key), dict):
                values[key] = {**values[key], **value}
            else:
                values[key] = value
    return parse(values)


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
