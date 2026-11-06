from dataclasses import asdict
import numpy as np
import pytest
from PIL import Image
from src.config import Config, parse, load_config, resolve
from src.data.io import pairs, read_pair, read_image
from src.data.dataset import MaskDataset
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
        _, _, _, side = zoom(image, np.zeros(image.shape), [1, 1], crop)
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
