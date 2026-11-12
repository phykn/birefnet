from dataclasses import asdict

import pytest

from src.config import load_config, parse_config, resolve, save_config


@pytest.mark.parametrize(
    "key,value",
    [
        ("augment.lights", 0.2),
        ("train.grad_checkpoint", False),
        ("train.steps", "3000"),
        ("train.steps", True),
        ("train.steps", 2.5),
        ("data.valid_ratio", 1.5),
        ("train.lr", float("nan")),
        ("model.classes", 256),
        ("train.crop", 65),
        ("loss.ce", -1),
        ("loss.boundary", -1),
        ("loss.boundary_width", 0),
        ("loss.aux", -1),
        ("loss.label_blur", 11),
        ("loss.teacher", 1),
        ("loss.gce_q", 1.5),
        ("loss.teacher_start", 0),
    ],
)
def test_invalid(key, value):
    section, field = key.split(".")
    with pytest.raises(ValueError, match=key):
        parse_config({section: {field: value}})


def test_zero_weights_rejected():
    with pytest.raises(ValueError, match="loss.mask_weight"):
        parse_config({"loss": {"label_weight": 0, "mask_weight": 0}})


def test_override(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("train:\n  crop: 512\n")
    expected = asdict(load_config())
    expected["train"]["crop"] = 512
    assert asdict(load_config(p)) == expected
    assert resolve("weight/a").is_absolute()


def test_config_paths_use_project_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    save_config(load_config(), root / "config.yaml")
    (root / "override.yaml").write_text("train:\n  crop: 64\n")
    monkeypatch.setattr("src.config.ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert load_config("override.yaml").train.crop == 64
