import importlib
import shutil
import sys

import matplotlib
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src import run
from src.config import load_config, save_config
from src.data import split
from src.data.io import pair_files, read_image
from src.model import Output, checkpoint
from src.model.checkpoint import load_model, load_trained
from src.predict import predict
from tests.helpers import Tiny, write, write_labels


def use_model(monkeypatch, model=Tiny):
    monkeypatch.setattr(run, "build_model", lambda cfg: model())
    monkeypatch.setattr(checkpoint, "build_model", lambda cfg: model())


def small_config(data, steps=2):
    cfg = load_config()
    cfg.data.image_dir = str(data / "image")
    cfg.data.mask_dir = str(data / "mask")
    cfg.train.steps = cfg.train.val_every = cfg.train.save_every = steps
    cfg.train.warmup = 0
    cfg.train.accum = 1
    cfg.train.crop = 64
    cfg.loader.batch_size = 1
    cfg.loader.num_workers = 0
    return cfg


def train_small(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    return run.train(config=path, runs=tmp_path / "run", device=torch.device("cpu"))


def run_script(monkeypatch, name, *args):
    monkeypatch.setattr(sys, "argv", [name, *args])
    importlib.import_module(f"scripts.{name}").main()


def test_typo_before_model(tmp_path, monkeypatch):
    import run_train

    p = tmp_path / "bad.yaml"
    p.write_text("augment:\n  lights: 1\n")
    monkeypatch.setattr(sys, "argv", ["run_train.py", "--config", str(p)])
    with pytest.raises(ValueError, match="augment.lights"):
        run_train.main()


def test_train_predict_evaluate(tmp_path, monkeypatch, capsys):
    use_model(monkeypatch)
    data = tmp_path / "data"
    write(data)
    target = train_small(tmp_path, small_config(data))
    assert {p.name for p in (target / "weights").iterdir()} == {
        "last.pth",
        "last.train.pth",
        "best_miou.pth",
    }
    log = EventAccumulator(str(target / "logs")).Reload()
    assert len(log.Scalars("train/loss")) == 2 and len(log.Scalars("valid/miou")) == 1
    copied = tmp_path / "detached.pth"
    shutil.copy(target / "weights/best_miou.pth", copied)
    image = read_image(data / "image/sample_01.png")
    m = load_model(copied, torch.device("cpu"))
    a = predict(m, image, size=64)["classes"]
    load_trained(m, torch.load(target / "weights/best_miou.pth")["model"])
    np.testing.assert_array_equal(a, predict(m, image, size=64)["classes"])
    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    run_script(
        monkeypatch, "03_predict",
        "--weight", str(copied),
        "--image", str(data / "image/sample_01.png"),
        "--mask", str(data / "mask/sample_01.png"),
        "--tiles", "1", "2",
        "--class-id", "1",
        "--device", "cpu",
    )
    run_script(
        monkeypatch, "04_evaluate",
        "--weight", str(copied),
        "--images", str(data / "image"),
        "--masks", str(data / "mask"),
        "--device", "cpu",
    )
    out = capsys.readouterr().out
    assert "pairs: 4" in out and "pixel_accuracy:" in out


def test_train_with_labels(tmp_path, monkeypatch):
    use_model(monkeypatch)
    data = tmp_path / "data"
    write(data)

    def paint(mask):
        region = np.zeros(mask.shape, bool)
        region[100:110, 100:140] = True
        return region

    write_labels(data, paint)
    cfg = small_config(data)
    cfg.data.label_dir = str(data / "label")
    target = train_small(tmp_path, cfg)
    assert (target / "train.csv").read_text().startswith("image,mask,label")
    tags = set(EventAccumulator(str(target / "logs")).Reload().Tags()["scalars"])
    assert {"train/label_ce", "train/ce", "train/dice"} <= tags
    for name in ("label", "mask"):
        assert {f"valid/{name}_miou", f"valid/{name}_pixel_accuracy"} <= tags
        assert {f"valid/{name}_class_{i}_iou" for i in range(4)} <= tags
    assert "valid/miou" not in tags
    metrics = torch.load(target / "weights/best_miou.pth")["metrics"]
    assert "label_miou" in metrics and "mask_miou" in metrics


class HeadTiny(Tiny):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.conv_out1 = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))

    def forward(self, x):
        return Output([self.decoder.conv_out1(x)])


def test_class_heads_get_a_larger_lr(tmp_path, monkeypatch):
    use_model(monkeypatch, HeadTiny)
    data = tmp_path / "data"
    write(data)
    cfg = small_config(data, steps=1)
    target = train_small(tmp_path, cfg)
    groups = torch.load(target / "weights/last.train.pth")["optimizer"]["param_groups"]
    assert len(groups[0]["params"]) == 0 and len(groups[1]["params"]) == 2
    assert groups[1]["initial_lr"] == cfg.train.lr * cfg.train.head_lr_scale


def test_early_rejections(tmp_path, monkeypatch):
    def forbidden(cfg):
        raise AssertionError("model must not be built")

    monkeypatch.setattr(run, "build_model", forbidden)
    data = tmp_path / "data"
    write(data)
    target = tmp_path / "run"
    (target / "weights").mkdir(parents=True)
    save_config(small_config(data), target / "config.yaml")
    pairs = pair_files(data / "image", data / "mask")
    split.write_split_csv(split.split_pairs(pairs, 0.25), target)
    (data / "image/sample_01.png").unlink()
    (data / "mask/sample_01.png").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        run.train(resume=target / "weights/last.train.pth")


def test_train_and_resume_paths_use_project_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    write(root / "data")
    cfg = small_config(root / "data", steps=1)
    cfg.data.image_dir = "data/image"
    cfg.data.mask_dir = "data/mask"
    save_config(cfg, root / "config.yaml")
    monkeypatch.setattr("src.config.ROOT", root)
    use_model(monkeypatch)
    monkeypatch.chdir(tmp_path)
    target = run.train(runs="run", device=torch.device("cpu"))
    assert target.parent == root / "run"
    weights = target / "weights/last.train.pth"
    assert run.train(resume=weights.relative_to(root), device=torch.device("cpu")) == target
    assert torch.load(weights)["step"] == 1
