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
            "birefnet": {"num_classes": 4},
            "data": {
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "size": 8,
                "valid_ratio": 0.2,
            },
            "augment": {
                "brightness": 0.2, "contrast": 0.4,
            },
            "loader": {"batch": 1, "num_workers": 0, "pin_memory": False},
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
        "train_image": ["c.png", "b.png"],
        "train_mask": ["c.png", "b.png"],
        "valid_image": ["a.png"],
        "valid_mask": ["a.png"],
    }
    save_splits(splits, run_dir)
    loaded = load_splits(run_dir)
    cfg = OmegaConf.create(
        {
            "birefnet": {"num_classes": 4},
            "data": {
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "size": 8,
                "valid_ratio": 0.2,
            },
            "augment": {
                "brightness": 0.2, "contrast": 0.4,
            },
            "loader": {"batch": 1, "num_workers": 0, "pin_memory": False},
        }
    )

    train, valid, actual = build_data(cfg, loaded)

    assert actual == splits
    assert Path(train.dataset.data[0][0]).name == "c.png"
    assert Path(valid.dataset.data[0][0]).name == "a.png"


@pytest.mark.parametrize("workers, cuda", [(0, False), (2, True)])
def test_loader_defaults_follow_workers_and_device(monkeypatch, workers, cuda):
    from src.build.data import _loader

    monkeypatch.setattr("src.build.data.torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr("src.build.data.DataLoader", lambda **options: options)
    cfg = OmegaConf.create({"loader": {"batch": 1, "num_workers": workers}})
    options = _loader(cfg, [], shuffle=False)
    assert options["pin_memory"] is cuda
    if workers:
        assert options["persistent_workers"] is True
        assert options["prefetch_factor"] == 2
    else:
        assert "persistent_workers" not in options
        assert "prefetch_factor" not in options
