from dataclasses import asdict
import numpy as np
import pytest
from PIL import Image
from src.config import Config, parse, load_config, read_config, resolve, save_config
import random
import hashlib
from src.data.io import pairs, read_pair, read_image, read_sample
from src.data.dataset import MaskDataset, hide_boundaries
from src.data import split
from src.data.augment import zoom, edge_mask
from src.prepare import fit_tensor
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
        parse({section: {field: value}})


def test_override(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("train:\n  crop: 512\n")
    cfg = load_config(p)
    expected = asdict(load_config())
    expected["train"]["crop"] = 512
    assert asdict(cfg) == expected
    assert resolve("weight/a").is_absolute()


@pytest.mark.parametrize("reader", [load_config, read_config])
def test_config_paths_use_project_root(tmp_path, monkeypatch, reader):
    root = tmp_path / "project"
    root.mkdir()
    save_config(Config(), root / "config.yaml")
    (root / "override.yaml").write_text("train:\n  crop: 64\n")
    monkeypatch.setattr("src.config.ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert reader("override.yaml").train.crop == 64


def test_io(tmp_path):
    write(tmp_path)
    data = pairs(tmp_path / "image", tmp_path / "mask")
    assert len(data) == 4
    image, mask = read_pair(*data[0], 4)
    assert image.ndim == 2 and mask.max() == 3
    p = tmp_path / "deep.tif"
    Image.fromarray(np.array([[1000, 4000]], np.uint16)).save(p)
    assert read_image(p).tolist() == [[0, 255]]
    Image.fromarray(np.full((2, 2), 3000, np.uint16)).save(p)
    assert not read_image(p).any()
    Image.fromarray(np.full(image.shape, 7, np.uint8)).save(data[0][1])
    with pytest.raises(ValueError, match="sample_01"):
        read_pair(*data[0], 4)
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(data[0][1])
    with pytest.raises(ValueError, match="sample_01"):
        read_pair(*data[0], 4)
    Image.fromarray(np.zeros((*image.shape, 3), np.uint8)).save(data[0][1])
    with pytest.raises(ValueError, match="mode RGB"):
        read_pair(*data[0], 4)
    Image.fromarray(image).save(tmp_path / "image/sample_01.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        pairs(tmp_path / "image", tmp_path / "mask")
    (tmp_path / "image/sample_01.jpg").unlink()
    (tmp_path / "mask/sample_01.png").unlink()
    with pytest.raises(ValueError, match="sample_01"):
        pairs(tmp_path / "image", tmp_path / "mask")


def test_split_restore(tmp_path):
    write(tmp_path)
    data = pairs(tmp_path / "image", tmp_path / "mask")
    s = split.make(data, 0.25, 0)
    split.save(s, tmp_path)
    assert split.restore(data, split.load(tmp_path)) == s
    with pytest.raises(RuntimeError):
        split.restore(data[:-1], split.load(tmp_path))


def test_dataset_and_crop(tmp_path):
    write(tmp_path)
    data = pairs(tmp_path / "image", tmp_path / "mask")
    cfg = Config()
    cfg.train.crop = 1024
    cfg.augment.zoom = cfg.augment.gamma = [1, 1]
    for k in ("hide_edges", "brightness", "contrast", "light", "blur", "noise"):
        setattr(cfg.augment, k, 0)
    sample = MaskDataset(data, cfg)[0]
    image, _ = read_pair(*data[0], 4)
    expected, _ = fit_tensor(image, is_sem=True)
    variants = [np.rot90(expected, k, axes=(1, 2)) for k in range(4)]
    variants += [x[:, :, ::-1] for x in variants]
    assert any(np.max(abs(sample["image"] - v)) <= 1.0e-5 for v in variants)
    for crop in (512, 1024):
        _, _, side = zoom(image.shape, [1, 1], crop)
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
        _, m = edge_mask(x, mask)
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
    for key in ("hide_edges", "brightness", "contrast", "light", "blur", "noise"):
        setattr(cfg.augment, key, 0)


def test_three_folders(tmp_path):
    write(tmp_path)
    write_labels(tmp_path, lambda mask: np.ones(mask.shape, bool))
    data = pairs(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    assert len(data) == 4 and all(len(pair) == 3 for pair in data)
    (tmp_path / "label/sample_02.png").unlink()
    with pytest.raises(ValueError, match=r"missing_labels=\['sample_02'\]"):
        pairs(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    mask = np.asarray(Image.open(tmp_path / "mask/sample_01.png"))
    Image.fromarray(np.full(mask.shape, 9, np.uint8)).save(tmp_path / "label/sample_02.png")
    data = pairs(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    with pytest.raises(ValueError, match="sample_02"):
        read_sample(data[1], 4)


def test_label_split_columns(tmp_path):
    write(tmp_path)
    write_labels(tmp_path, lambda mask: mask > 0)
    two = pairs(tmp_path / "image", tmp_path / "mask")
    three = pairs(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    s2, s3 = split.make(two, 0.25, 0), split.make(three, 0.25, 0)
    for name in split.NAMES:
        assert [p[0] for p in s2[name]] == [p[0] for p in s3[name]]
    split.save(s3, tmp_path)
    assert (tmp_path / "train.csv").read_text().splitlines()[0] == "image,mask,label"
    assert split.restore(three, split.load(tmp_path)) == s3
    with pytest.raises(RuntimeError, match="label"):
        split.restore(two, split.load(tmp_path))


def test_label_crop_and_shared_transforms(tmp_path):
    write(tmp_path)
    write_labels(tmp_path, lambda mask: mask > 0)
    data = pairs(tmp_path / "image", tmp_path / "mask", tmp_path / "label")
    cfg = Config()
    cfg.model.classes = 4
    cfg.augment.hide_edges = 0.5
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


def test_hide_boundaries():
    mask = np.zeros((40, 40), np.int64)
    mask[:, 20:] = 1
    mask[0, 0] = 255
    hidden = hide_boundaries(mask, 4)
    assert (hidden[:, 16:24] == 255).all()
    assert (hidden[:, :16] == mask[:, :16]).all() and (hidden[:, 24:] == 1).all()


def test_dataset_without_labels_is_unchanged(tmp_path):
    # Fingerprints of samples drawn before label support was added.
    expected = [
        (178013.67112375144, 484346.7784190724, "e97bba48cd932dc6", "9302841f99219178"),
        (177555.0612605177, 484556.9720689157, "869c69034d69615e", "6fe447c6e220ffe3"),
        (178539.07632584032, 484559.85283645417, "70f0cc8aae8be723", "880768fb36f3cb45"),
    ]
    write(tmp_path)
    data = pairs(tmp_path / "image", tmp_path / "mask")
    for index, (total, square, mask_hash, cut_hash) in enumerate(expected):
        random.seed(index)
        np.random.seed(index)
        sample = MaskDataset(data, Config())[index]
        image = sample["image"].astype(np.float64)
        assert image.sum() == pytest.approx(total, abs=1e-2)
        assert (image**2).sum() == pytest.approx(square, abs=1e-2)
        assert hashlib.sha256(sample["mask"].tobytes()).hexdigest()[:16] == mask_hash
        assert hashlib.sha256(sample["cut"].tobytes()).hexdigest()[:16] == cut_hash
        assert "label" not in sample


def test_zero_weights_rejected():
    with pytest.raises(ValueError, match="loss.mask_weight"):
        parse({"loss": {"label_weight": 0, "mask_weight": 0}})
