from dataclasses import asdict
import numpy as np
import pytest
from PIL import Image
from src.config import load_config, parse_config, resolve, save_config
import random
from src.data.io import pair_files, read_image, read_sample
from src.data.dataset import MaskDataset
from src.data import split
from src.data.augment import hide_edges, pick_crop
from src.data.model_input import fit_image
from tests.sample_data import write


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
    ],
)
def test_invalid(key, value):
    section, field = key.split(".")
    with pytest.raises(ValueError, match=key):
        parse_config({section: {field: value}})


def test_override(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("train:\n  crop: 512\n")
    cfg = load_config(p)
    expected = asdict(load_config())
    expected["train"]["crop"] = 512
    assert asdict(cfg) == expected
    assert resolve("weight/a").is_absolute()


def test_config_paths_use_project_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    save_config(load_config(), root / "config.yaml")
    (root / "override.yaml").write_text("train:\n  crop: 64\n")
    monkeypatch.setattr("src.config.ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert load_config("override.yaml").train.crop == 64


def test_io(tmp_path):
    write(tmp_path)
    (tmp_path / "image/.DS_Store").write_bytes(b"")
    (tmp_path / "image/._sample_01.png").write_bytes(b"")
    data = pair_files(tmp_path / "image", tmp_path / "mask")
    assert len(data) == 4
    image, mask = read_sample(data[0], 4)
    assert image.ndim == 2 and mask.max() == 3
    p = tmp_path / "deep.tif"
    Image.fromarray(np.array([[0, 32768, 65535]], np.uint16)).save(p)
    assert read_image(p).tolist() == [[0, 128, 255]]
    Image.fromarray(np.full(image.shape, 7, np.uint8)).save(data[0][1])
    with pytest.raises(ValueError, match="sample_01"):
        read_sample(data[0], 4)
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(data[0][1])
    with pytest.raises(ValueError, match="sample_01"):
        read_sample(data[0], 4)
    Image.fromarray(np.zeros((*image.shape, 3), np.uint8)).save(data[0][1])
    with pytest.raises(ValueError, match="mode RGB"):
        read_sample(data[0], 4)
    Image.fromarray(image).save(tmp_path / "image/sample_01.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        pair_files(tmp_path / "image", tmp_path / "mask")
    (tmp_path / "image/sample_01.jpg").unlink()
    (tmp_path / "mask/sample_01.png").unlink()
    with pytest.raises(ValueError, match="sample_01"):
        pair_files(tmp_path / "image", tmp_path / "mask")


def test_split_restore(tmp_path):
    write(tmp_path)
    data = pair_files(tmp_path / "image", tmp_path / "mask")
    s = split.split_pairs(data, 0.25, 0)
    split.write_split_csv(s, tmp_path)
    assert split.read_split_csv(data, tmp_path) == s
    with pytest.raises(RuntimeError):
        split.read_split_csv(data[:-1], tmp_path)


def test_dataset_and_crop(tmp_path):
    write(tmp_path)
    data = pair_files(tmp_path / "image", tmp_path / "mask")
    cfg = load_config()
    cfg.train.crop = 1024
    cfg.augment.zoom = [1, 1]
    for k in ("gamma", "shading", "blur", "noise", "hide_edges"):
        getattr(cfg.augment, k).p = 0
    sample = MaskDataset(data, cfg)[0]
    image, _ = read_sample(data[0], 4)
    expected, _ = fit_image(image)
    variants = [np.rot90(expected, k, axes=(1, 2)) for k in range(4)]
    variants += [x[:, :, ::-1] for x in variants]
    assert any(np.max(abs(sample["image"] - v)) <= 1.0e-5 for v in variants)
    for crop in (512, 1024):
        _, _, side = pick_crop(image.shape, [1, 1], crop)
        assert crop / side == 1024 / max(image.shape)
    cfg.train.crop = 512
    assert MaskDataset(data, cfg)[0]["image"].shape == (3, 512, 512)
    gray = np.arange(80 * 120, dtype=np.uint8).reshape(80, 120)
    Image.fromarray(gray).save(data[0][0])
    Image.fromarray(np.zeros_like(gray)).save(data[0][1])
    cfg.train.crop = 1024
    assert (MaskDataset(data, cfg)[0]["mask"] == 255).any()


def test_hidden_edges():
    x = np.zeros((200, 200), np.uint8)
    mask = np.ones_like(x)
    for _ in range(10):
        _, m = hide_edges(x, mask)
        assert (m != 255).sum() >= mask.size / 2


def write_labels(root, paint):
    """Write labels equal to the mask where ``paint`` is true and 255 elsewhere."""
    (root / "label").mkdir()
    for mask_path in sorted((root / "mask").iterdir()):
        mask = np.asarray(Image.open(mask_path))
        Image.fromarray(np.where(paint(mask), mask, 255).astype(np.uint8)).save(
            root / "label" / mask_path.name
        )


def quiet_augment(cfg):
    for key in ("shading", "blur", "noise", "hide_edges"):
        getattr(cfg.augment, key).p = 0


def test_three_folders(tmp_path):
    write(tmp_path)
    write_labels(tmp_path, lambda mask: np.ones(mask.shape, bool))
    data = pair_files(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    assert len(data) == 4 and all(len(pair) == 3 for pair in data)
    (tmp_path / "label/sample_02.png").unlink()
    with pytest.raises(ValueError, match=r"missing_labels=\['sample_02'\]"):
        pair_files(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    mask = np.asarray(Image.open(tmp_path / "mask/sample_01.png"))
    Image.fromarray(np.full(mask.shape, 9, np.uint8)).save(tmp_path / "label/sample_02.png")
    data = pair_files(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    with pytest.raises(ValueError, match="sample_02"):
        read_sample(data[1], 4)


def test_label_split_columns(tmp_path):
    write(tmp_path)
    write_labels(tmp_path, lambda mask: mask > 0)
    two = pair_files(tmp_path / "image", tmp_path / "mask")
    three = pair_files(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    s2, s3 = split.split_pairs(two, 0.25, 0), split.split_pairs(three, 0.25, 0)
    for name in ("train", "valid"):
        assert [p[0] for p in s2[name]] == [p[0] for p in s3[name]]
    split.write_split_csv(s3, tmp_path)
    assert (tmp_path / "train.csv").read_text().splitlines()[0] == "image,mask,label"
    assert split.read_split_csv(three, tmp_path) == s3
    with pytest.raises(RuntimeError, match="label"):
        split.read_split_csv(two, tmp_path)


def test_label_crop_and_shared_transforms(tmp_path):
    write(tmp_path)
    write_labels(tmp_path, lambda mask: mask > 0)
    data = pair_files(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    cfg = load_config()
    cfg.model.classes = 4
    cfg.augment.hide_edges.p = 0.5
    for seed in range(10):
        random.seed(seed)
        np.random.seed(seed)
        sample = MaskDataset(data, cfg)[seed % len(data)]
        painted = sample["label"] != 255
        assert (sample["label"][painted] == sample["mask"][painted]).all()
    single = np.full((256, 256), 255, np.uint8)
    single[200, 30] = 2
    Image.fromarray(single).save(data[0][2])
    quiet_augment(cfg)
    cfg.augment.zoom = [2.0, 2.0]
    for seed in range(20):
        random.seed(seed)
        np.random.seed(seed)
        assert (MaskDataset(data, cfg)[0]["label"] != 255).any()


def test_zero_weights_rejected():
    with pytest.raises(ValueError, match="loss.mask_weight"):
        parse_config({"loss": {"label_weight": 0, "mask_weight": 0}})
