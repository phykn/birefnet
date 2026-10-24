from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.build.data import build as build_data
from src.data.pairs import index
from src.data.split import load as load_splits
from src.data.split import save as save_splits


def test_index_by_stem_maps_stem_to_path():
    paths = ["/data/image/a.png", "/data/image/b.jpg"]
    result = index(paths)
    assert result == {"a": "/data/image/a.png", "b": "/data/image/b.jpg"}


def test_index_by_stem_raises_on_duplicate():
    paths = ["/x/a.png", "/y/a.jpg"]
    with pytest.raises(ValueError, match="Duplicate stem"):
        index(paths)


def test_index_by_stem_empty():
    assert index([]) == {}


def test_build_loaders_rejects_unpaired_stems(tmp_path):
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.new("RGB", (4, 4)).save(image_dir / "sample.png")
    cfg = OmegaConf.create(
        {
            "model": {"classes": 4},
            "data": {
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "size": 8,
                "valid_ratio": 0.2,
            },
            "augment": {
                "brightness": 0.2, "contrast": 0.4,
            },
            "loader": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
        }
    )
    with pytest.raises(ValueError, match="stems do not match"):
        build_data(cfg)


def test_build_loaders_restores_saved_split_membership(tmp_path):
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    run_dir = tmp_path / "run"
    image_dir.mkdir()
    mask_dir.mkdir()
    run_dir.mkdir()
    for name in ["a", "b", "c"]:
        Image.new("RGB", (4, 4)).save(image_dir / f"{name}.png")
        Image.new("L", (4, 4)).save(mask_dir / f"{name}.png")

    splits = {
        "train": [("c.png", "c.png"), ("b.png", "b.png")],
        "valid": [("a.png", "a.png")],
    }
    save_splits(splits, run_dir)
    loaded = load_splits(run_dir)
    cfg = OmegaConf.create(
        {
            "model": {"classes": 4},
            "data": {
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "size": 8,
                "valid_ratio": 0.2,
            },
            "augment": {
                "brightness": 0.2, "contrast": 0.4,
            },
            "loader": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
        }
    )

    train, actual = build_data(cfg, loaded)

    assert actual["train"] == train.dataset.pairs
    for name in ("train", "valid"):
        assert [(Path(image).name, Path(mask).name) for image, mask in actual[name]] == splits[name]


@pytest.mark.parametrize("workers, cuda", [(0, False), (2, True)])
def test_loader_defaults_follow_workers_and_device(monkeypatch, workers, cuda):
    from src.build.data import _loader

    monkeypatch.setattr("src.build.data.torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr("src.build.data.DataLoader", lambda **options: options)
    cfg = OmegaConf.create({"loader": {"batch_size": 1, "num_workers": workers}})
    options = _loader(cfg, [1])
    assert options["pin_memory"] is cuda
    if workers:
        assert options["persistent_workers"] is True
        assert options["prefetch_factor"] == 2
    else:
        assert "persistent_workers" not in options
        assert "prefetch_factor" not in options


def test_loader_options_come_from_config(monkeypatch):
    from src.build.data import _loader

    monkeypatch.setattr("src.build.data.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.build.data.DataLoader", lambda **options: options)
    cfg = OmegaConf.create({"loader": {"batch_size": 2, "num_workers": 3, "prefetch_factor": 4,
                                       "persistent_workers": False, "pin_memory": False}})
    options = _loader(cfg, [1, 2])
    assert (options["batch_size"], options["num_workers"], options["prefetch_factor"]) == (2, 3, 4)
    assert options["persistent_workers"] is False and options["pin_memory"] is False


@pytest.mark.parametrize("drop_last", [True, False])
def test_drop_last_comes_from_config(monkeypatch, drop_last):
    from src.build.data import _loader

    monkeypatch.setattr("src.build.data.DataLoader", lambda **options: options)
    cfg = OmegaConf.create({"loader": {"batch_size": 2, "num_workers": 0, "drop_last": drop_last}})
    assert _loader(cfg, [1, 2, 3])["drop_last"] is drop_last


def test_drop_last_rejects_a_training_set_smaller_than_one_batch(monkeypatch):
    from src.build.data import _loader

    cfg = OmegaConf.create({"loader": {"batch_size": 4, "num_workers": 0, "drop_last": True}})
    with pytest.raises(ValueError, match="drop_last"):
        _loader(cfg, [1, 2, 3])


@pytest.mark.parametrize("name,value", [
    ("batch_size", 1.5), ("batch_size", True),
    ("num_workers", 1.5), ("num_workers", True),
    ("prefetch_factor", 1.5), ("prefetch_factor", True),
    ("pin_memory", "false"), ("persistent_workers", "false"),
])
def test_loader_rejects_settings_that_would_be_silently_coerced(name, value):
    from src.build.data import _loader

    cfg = OmegaConf.create({"loader": {"batch_size": 1, "num_workers": 2, name: value}})
    with pytest.raises(ValueError, match=name):
        _loader(cfg, [1, 2])


@pytest.mark.parametrize("weight_decay", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_trainer_build_rejects_invalid_weight_decay(tmp_path, weight_decay):
    import torch
    from src.build.trainer import build as build_trainer

    model = torch.nn.Linear(2, 4)
    model.num_classes = 4
    model.list_trainable = lambda: list(model.parameters())
    cfg = OmegaConf.create({
        "model": {"classes": 4},
        "data": {},
        "loss": {"ce": 1, "dice": 1, "boundary": 0.5,
                 "boundary_width": 3, "aux": 1},
        "train": {"lr": 0.01, "min_lr": 0.001, "warmup": 0,
                  "steps": 10, "weight_decay": weight_decay,
                  "grad_clip": 1, "accum": 1},
    })

    with pytest.raises(ValueError, match="weight_decay"):
        build_trainer(cfg, model, [], [], tmp_path)
