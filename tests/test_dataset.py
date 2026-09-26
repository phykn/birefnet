import numpy as np
import pytest
from PIL import Image

from src.data.augment import crop
from src.data.dataset import MaskDataset
from src.data.image import read_image, read_mask


def _write_pair(tmp_path, palette=False):
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    mask = np.tile(np.arange(20, dtype=np.uint8) % 4, (12, 1))
    mask[0, 0] = 255
    image_path = tmp_path / "sample.png"
    mask_path = tmp_path / "sample_mask.png"
    Image.fromarray(image).save(image_path)
    label_image = Image.fromarray(mask)
    if palette:
        label_image = label_image.convert("P")
        label_image.putpalette([channel for index in range(256) for channel in (index, 255-index, 127)])
    label_image.save(mask_path)
    return image_path, mask_path, mask


def test_loader_uses_rgb_channel_order(tmp_path):
    image_path, _, _ = _write_pair(tmp_path)
    assert read_image(str(image_path))[0, 0].tolist() == [255, 0, 0]


@pytest.mark.parametrize("palette", [False, True])
def test_class_indices_survive_palette_loading_and_letterbox(tmp_path, palette):
    image_path, mask_path, labels = _write_pair(tmp_path, palette)
    np.testing.assert_array_equal(read_mask(str(mask_path)), labels)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32)[0]
    assert sample["weak"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert sample["mask"].dtype == np.int64
    assert set(np.unique(sample["mask"])) == {0, 1, 2, 3, 255}
    assert sample["valid"].shape == sample["cut"].shape == (1, 32, 32)
    assert np.all(sample["mask"][sample["valid"][0] == 0] == 255)
    assert not sample["cut"].any()


def test_training_views_share_geometry(tmp_path):
    image_path, mask_path, _ = _write_pair(tmp_path)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32, train=True, global_prob=1)[0]
    assert sample["weak"].shape == sample["strong"].shape
    assert sample["mask"].shape == sample["valid"].shape[1:]
    assert sample["cut"].shape == sample["valid"].shape


def test_rejects_invalid_indices_and_rgb_masks(tmp_path):
    image_path, mask_path, mask = _write_pair(tmp_path)
    mask[1, 1] = 4
    Image.fromarray(mask).save(mask_path)
    with pytest.raises(ValueError, match="invalid class indices"):
        MaskDataset([(str(image_path), str(mask_path))], size=32)[0]
    Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(mask_path)
    with pytest.raises(ValueError, match="class indices"):
        read_mask(str(mask_path))


def test_crop_marks_only_edges_cut_from_source(monkeypatch):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    picks = iter((0, 15))
    monkeypatch.setattr("random.randrange", lambda *_: next(picks))
    _, _, cut = crop(image, mask, size=10, global_prob=0, boundary_prob=0)
    assert not cut[0, 1:-1].any()
    assert cut[-1].all() and cut[:, 0].all() and cut[:, -1].all()


def test_boundary_crop_detects_adjacent_nonbackground_classes(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.ones((20, 20), dtype=np.uint8)
    mask[:, 10:] = 2
    monkeypatch.setattr("random.randrange", lambda *_: 0)
    _, labels, _ = crop(image, mask, 6, 0, 1)
    assert set(np.unique(labels)) == {1, 2}
