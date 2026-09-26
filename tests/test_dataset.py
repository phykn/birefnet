import numpy as np
import pytest
from PIL import Image

from src.data.augment import crop, edge_mask, jitter
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
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert sample["mask"].dtype == np.int64
    assert set(np.unique(sample["mask"])) == {0, 1, 2, 3, 255}
    assert sample["valid"].shape == sample["cut"].shape == (1, 32, 32)
    assert np.all(sample["mask"][sample["valid"][0] == 0] == 255)
    assert not sample["cut"].any()


def test_training_image_matches_label_geometry(tmp_path):
    image_path, mask_path, _ = _write_pair(tmp_path)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32, min_crop_size=32, train=True, crop_prob=0)[0]
    assert sample["image"].shape == (3, 32, 32)
    assert set(sample) == {"image", "mask", "valid", "cut"}
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
    _, _, cut = crop(image, mask, size=10, crop_prob=1, min_size=10)
    assert not cut[0, 1:-1].any()
    assert cut[-1].all() and cut[:, 0].all() and cut[:, -1].all()


def test_crop_position_does_not_depend_on_mask_labels(monkeypatch):
    image = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    mask = np.zeros((20, 20), dtype=np.uint8)
    boundary_mask = mask.copy()
    boundary_mask[:, 10:] = 2
    monkeypatch.setattr("random.randrange", lambda limit: limit // 2)
    plain, _, _ = crop(image, mask, 6, 1, 6)
    with_boundary, labels, _ = crop(image, boundary_mask, 6, 1, 6)
    np.testing.assert_array_equal(plain, with_boundary)
    np.testing.assert_array_equal(labels, boundary_mask[7:13, 7:13])


@pytest.mark.parametrize("prob, shape", [(0.0, (20, 30)), (0.3, (20, 30)),
                                        (0.5, (10, 10)), (1.0, (10, 10))])
def test_crop_probability_selects_crop_instead_of_full_image(monkeypatch, prob, shape):
    monkeypatch.setattr("random.random", lambda: 0.4)
    image = np.zeros((20, 30, 3), np.uint8)
    mask = np.zeros((20, 30), np.uint8)
    result, _, _ = crop(image, mask, 10, prob, 10)
    assert result.shape[:2] == shape


def test_sem_jitter_precedes_channels_and_resize_precedes_padding(tmp_path, monkeypatch):
    from src.prepare.convert import convert
    from src.prepare.fit import fit_image, fit_mask

    image_path, mask_path, labels = _write_pair(tmp_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    values = iter((70,))

    def jitter_gray(image, limits):
        assert image.ndim == 2
        return np.full_like(image, next(values))

    monkeypatch.setattr("src.data.dataset.jitter", jitter_gray)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32,
                         train=True, is_sem=True, crop_prob=1, min_crop_size=32)[0]
    expected, valid, fit = fit_image(convert(np.full((12, 20), 70, np.uint8), True), size=32)
    expected_labels = fit_mask(labels, fit)[0].astype(np.int64)
    expected_labels[valid[0] == 0] = 255
    np.testing.assert_array_equal(sample["image"], expected)
    np.testing.assert_array_equal(sample["mask"], expected_labels)
    assert (fit.dst_h, fit.dst_w, fit.top) == (19, 32, 6)
    assert np.all(sample["mask"][:6] == 255)
    assert not sample["valid"][0, :6].any()


def test_sem_validation_matches_inference_preprocessing(tmp_path):
    from src.prepare.fit import fit_tensor

    image_path, mask_path, _ = _write_pair(tmp_path)
    dataset = MaskDataset([(str(image_path), str(mask_path))], size=32,
                          is_sem=True, masking_prob=1)
    first, second = dataset[0], dataset[0]
    expected, _ = fit_tensor(read_image(str(image_path)), size=32, is_sem=True)
    np.testing.assert_array_equal(first["image"], expected)
    np.testing.assert_array_equal(first["image"], second["image"])
    assert set(first) == {"image", "mask", "valid", "cut"}


def test_edge_mask_preserves_sparse_roi():
    import random

    state = random.getstate()
    random.seed(17)
    try:
        mask = np.full((40, 48), 255, np.int64)
        mask[:10, :20] = 2
        image = np.full((40, 48, 3), 32, np.uint8)
        changed = False
        for _ in range(50):
            result, labels = edge_mask(image, mask)
            assert np.count_nonzero(labels != 255) >= 100
            np.testing.assert_array_equal(result[10:30, 12:36], image[10:30, 12:36])
            hidden = np.any(result != image, axis=-1)
            assert np.all(labels[hidden] == 255)
            np.testing.assert_array_equal(labels[~hidden], mask[~hidden])
            changed |= bool(hidden.any())
        assert changed
        assert np.all(image == 32)
    finally:
        random.setstate(state)


def test_edge_mask_empty_roi_and_bounded_retry(monkeypatch):
    image = np.zeros((16, 16, 3), np.uint8)
    mask = np.full((16, 16), -1, np.int64)
    assert edge_mask(image, mask, -1)[1] is mask
    mask[0, 0] = 0
    calls = []

    def maximum(low, high):
        calls.append(high)
        return high

    monkeypatch.setattr("random.randint", maximum)
    assert edge_mask(image, mask, -1)[1] is mask
    assert len(calls) == 32 * 4


def test_edge_mask_updates_dataset_valid(tmp_path, monkeypatch):
    image_path, mask_path, _ = _write_pair(tmp_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    monkeypatch.setattr("random.randint", lambda low, high: max(low, high // 2))
    sample = MaskDataset([(str(image_path), str(mask_path))], size=20,
                         train=True, is_sem=True, crop_prob=1,
                         masking_prob=1, min_crop_size=20)[0]
    # The 12x20 image is centered with four padding rows; the first image row is hidden.
    assert np.all(sample["mask"][4] == 255)
    assert not sample["valid"][0, 4].any()
    assert sample["valid"].sum() >= (12 * 20 - 1) / 2


def test_full_size_crop_keeps_native_pixels(tmp_path, monkeypatch):
    from src.prepare.convert import normalize

    gray = np.arange(24 * 40, dtype=np.uint8).reshape(24, 40)
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(gray).save(image_path)
    Image.fromarray((gray % 4).astype(np.uint8)).save(mask_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    monkeypatch.setattr("random.randrange", lambda limit: limit // 2)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=16, min_crop_size=16, train=True,
                         is_sem=True, crop_prob=1,
                         brightness=0, contrast=0)[0]
    patch = gray[4:20, 12:28]
    expected = normalize(np.repeat(patch[..., None], 3, axis=-1))[..., 0]
    np.testing.assert_array_equal(sample["image"][0], expected)
    np.testing.assert_array_equal(sample["mask"], patch % 4)


@pytest.mark.parametrize("side", [256, 637, 1024])
def test_random_crop_size_range(monkeypatch, side):
    def pick(low, high):
        assert (low, high) == (256, 1024)
        return side

    monkeypatch.setattr("random.randint", pick)
    monkeypatch.setattr("random.randrange", lambda limit: limit // 2)
    mask = np.indices((1200, 1400))[1].astype(np.uint16)
    image = np.repeat(mask[..., None], 3, axis=2)
    patch, labels, cut = crop(image, mask, 1024, 1, 256)
    assert patch.shape == (side, side, 3)
    np.testing.assert_array_equal(patch[..., 0], labels)
    assert cut[0].all() and cut[-1].all()
    assert cut[:, 0].all() and cut[:, -1].all()


def test_crop_clips_to_source_without_padding(monkeypatch):
    monkeypatch.setattr("random.randint", lambda low, high: low)
    mask = np.zeros((100, 400), np.uint8)
    image = np.zeros((100, 400, 3), np.uint8)
    patch, labels, cut = crop(image, mask, 1024, 1, 256)
    assert patch.shape == (100, 256, 3)
    assert labels.shape == cut.shape == (100, 256)
    assert not cut[0, 1:-1].any()
    assert not cut[-1, 1:-1].any()


@pytest.mark.parametrize("bright, contrast, expected", [(0.2, 0.0, 115), (-0.2, 0.0, 13),
                                                       (0.0, 0.4, 39), (0.0, -0.4, 89)])
def test_jitter_samples_both_directions(monkeypatch, bright, contrast, expected):
    values = iter((bright, contrast))
    monkeypatch.setattr("random.uniform", lambda low, high: next(values))
    result = jitter(np.full((4, 4), 64, np.uint8), (0.2, 0.4))
    assert np.all(result == expected)
