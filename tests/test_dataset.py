import numpy as np
import pytest
from PIL import Image

from src.data.augment import blur, edge_mask, gamma, illumination, jitter, noise, zoom
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


@pytest.mark.parametrize("values, expected", [
    ([0, 1000, 30000, 65535], [0, 4, 117, 255]),
    ([100, 1050, 2100, 4100], [0, 61, 128, 255]),  # 12-bit data in a 16-bit file
    ([500, 500, 500, 500], [0, 0, 0, 0]),
])
def test_16_bit_images_stretch_between_their_own_extremes(tmp_path, values, expected):
    path = tmp_path / "deep.tif"
    Image.fromarray(np.array([values], dtype=np.uint16)).save(path)
    image = read_image(str(path))
    assert image.dtype == np.uint8 and image.shape == (1, 4, 3)
    assert image[..., 0].tolist() == [expected]
    assert np.array_equal(image[..., 0], image[..., 2])


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
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32)[0]
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


def test_zoom_marks_only_edges_cut_from_source(monkeypatch):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    picks = iter((0, 15))
    monkeypatch.setattr("random.randint", lambda *_: next(picks))
    _, _, cut, side = zoom(image, mask, (3.0, 3.0))
    assert side == 10
    assert not cut[0, 1:-1].any()
    assert cut[-1].all() and cut[:, 0].all() and cut[:, -1].all()


def test_zoom_position_does_not_depend_on_mask_labels(monkeypatch):
    image = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    mask = np.zeros((20, 20), dtype=np.uint8)
    boundary_mask = mask.copy()
    boundary_mask[:, 10:] = 2
    monkeypatch.setattr("random.randint", lambda low, high: high // 2)
    plain, _, _, _ = zoom(image, mask, (10 / 3, 10 / 3))
    with_boundary, labels, _, _ = zoom(image, boundary_mask, (10 / 3, 10 / 3))
    np.testing.assert_array_equal(plain, with_boundary)
    np.testing.assert_array_equal(labels, boundary_mask[7:13, 7:13])


@pytest.mark.parametrize("scale, side, shape", [(0.5, 2800, (1200, 1400)), (1.0, 1400, (1200, 1400)),
                                               (2.0, 700, (700, 700)), (4.0, 350, (350, 350))])
def test_zoom_window_follows_magnification(monkeypatch, scale, side, shape):
    monkeypatch.setattr("random.randint", lambda low, high: high // 2)
    mask = np.indices((1200, 1400))[1].astype(np.uint16)
    image = np.repeat(mask[..., None], 3, axis=2)
    patch, labels, cut, picked = zoom(image, mask, (scale, scale))
    assert picked == side
    assert patch.shape[:2] == labels.shape == cut.shape == shape
    np.testing.assert_array_equal(patch[..., 0], labels)
    assert cut.any() == (shape != (1200, 1400))


def test_zoom_out_pads_and_excludes_padding(tmp_path, monkeypatch):
    image_path, mask_path, _ = _write_pair(tmp_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32,
                         zoom=(0.5, 0.5))[0]
    # Half of the full fit (19x32): the 12x20 image becomes 10x16 in the middle.
    outside = np.ones((32, 32), bool)
    outside[11:21, 8:24] = False
    assert not sample["valid"][0][outside].any()
    assert np.all(sample["mask"][outside] == 255)
    assert sample["valid"][0, 11:21, 8:24].sum() >= 10 * 16 - 1


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
                         is_sem=True)[0]
    expected, valid, fit = fit_image(convert(np.full((12, 20), 70, np.uint8), True), size=32)
    expected_labels = fit_mask(labels, fit)[0].astype(np.int64)
    expected_labels[valid[0] == 0] = 255
    np.testing.assert_array_equal(sample["image"], expected)
    np.testing.assert_array_equal(sample["mask"], expected_labels)
    assert (fit.dst_h, fit.dst_w, fit.top) == (19, 32, 6)
    assert np.all(sample["mask"][:6] == 255)
    assert not sample["valid"][0, :6].any()


def test_training_without_augmentation_matches_inference_preprocessing(tmp_path, monkeypatch):
    from src.prepare.fit import fit_tensor

    image_path, mask_path, _ = _write_pair(tmp_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32,
                         is_sem=True, brightness=0, contrast=0)[0]
    expected, _ = fit_tensor(read_image(str(image_path)), size=32, is_sem=True)
    np.testing.assert_array_equal(sample["image"], expected)


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
                         is_sem=True, masking_prob=1)[0]
    # The 12x20 image is centered with four padding rows; the first image row is hidden.
    assert np.all(sample["mask"][4] == 255)
    assert not sample["valid"][0, 4].any()
    assert sample["valid"].sum() >= (12 * 20 - 1) / 2


def test_full_size_crop_keeps_native_pixels(tmp_path, monkeypatch):
    from src.prepare.convert import convert, normalize

    gray = np.arange(24 * 40, dtype=np.uint8).reshape(24, 40)
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(gray).save(image_path)
    Image.fromarray((gray % 4).astype(np.uint8)).save(mask_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    monkeypatch.setattr("random.randint", lambda low, high: high // 2)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=16,
                         is_sem=True, zoom=(2.5, 2.5),
                         brightness=0, contrast=0)[0]
    patch = gray[4:20, 12:28]
    expected = normalize(convert(patch, is_sem=True))[..., 0]
    np.testing.assert_array_equal(sample["image"][0], expected)
    np.testing.assert_array_equal(sample["mask"], patch % 4)


@pytest.mark.parametrize("bright, contrast, expected", [(0.2, 0.0, 115), (-0.2, 0.0, 13),
                                                       (0.0, 0.4, 39), (0.0, -0.4, 89)])
def test_jitter_samples_both_directions(monkeypatch, bright, contrast, expected):
    values = iter((bright, contrast))
    monkeypatch.setattr("random.uniform", lambda low, high: next(values))
    result = jitter(np.full((4, 4), 64, np.uint8), (0.2, 0.4))
    assert np.all(result == expected)


def test_gamma_bends_midtones_and_keeps_ends(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda low, high: np.log(2.0))
    result = gamma(np.array([0, 128, 255], np.uint8), (2.0, 2.0))
    assert result.tolist() == [0, 64, 255]


def test_illumination_ramps_across_the_image(monkeypatch):
    values = iter((0.0, 0.3))
    monkeypatch.setattr("random.uniform", lambda low, high: next(values))
    result = illumination(np.full((4, 5), 100, np.uint8), 0.3)
    # Angle 0 ramps left to right: 0.7x on the left edge, 1.3x on the right.
    assert np.all(result[:, 0] == 70) and np.all(result[:, -1] == 130)
    assert np.all(result[:, 2] == 100)


def test_blur_softens_a_step_edge(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda low, high: 1.0)
    image = np.zeros((9, 9), np.uint8)
    image[:, 5:] = 200
    result = blur(image, 1.0)
    assert 0 < result[4, 4] < result[4, 5] < 200
    assert result[4, 0] == 0 and result[4, -1] == 200


def test_noise_adds_grain_of_the_drawn_strength(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda low, high: 0.05)
    result = noise(np.full((200, 200), 128, np.uint8), 0.05).astype(np.float32)
    assert abs(result.mean() - 128) < 1
    assert abs(result.std() - 0.05 * 255) < 0.5


def test_zero_blur_and_noise_keep_the_image():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    np.testing.assert_array_equal(blur(image, 0.0), image)
    np.testing.assert_array_equal(noise(image, 0.0), image)


@pytest.mark.parametrize("name, value", [("zoom", (0.0, 2.0)), ("zoom", (2.0, 1.0)),
                                         ("zoom", (0.5, float("inf"))), ("gamma", (-1.0, 1.0)),
                                         ("gamma", (float("nan"), 1.0)), ("illumination", 1.0),
                                         ("illumination", -0.1),
                                         ("blur", -1.0), ("blur", float("inf")), ("noise", 1.0),
                                         ("noise", -0.1)])
def test_training_dataset_rejects_bad_photometric_ranges(name, value):
    with pytest.raises(ValueError, match=name):
        MaskDataset([], **{name: value})


@pytest.mark.parametrize("name", ["brightness", "contrast"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_training_dataset_rejects_nonfinite_jitter(name, value):
    with pytest.raises(ValueError, match=name):
        MaskDataset([], **{name: value})


def test_hidden_edges_enter_the_z_score(tmp_path, monkeypatch):
    from src.prepare.convert import convert
    from src.prepare.fit import fit_image

    image_path, mask_path, _ = _write_pair(tmp_path)
    monkeypatch.setattr("src.data.dataset.flip", lambda *items: items)
    hidden = []

    def hide_left(image, mask, ignore_index):
        assert image.ndim == 2  # still the raw gray image, before the z-score
        image, mask = image.copy(), mask.copy()
        image[:, :5] = 255
        mask[:, :5] = ignore_index
        hidden.append(image)
        return image, mask

    monkeypatch.setattr("src.data.dataset.edge_mask", hide_left)
    sample = MaskDataset([(str(image_path), str(mask_path))], size=32, is_sem=True,
                         masking_prob=1, brightness=0, contrast=0)[0]
    expected, _, _ = fit_image(convert(hidden[0], is_sem=True), size=32)
    np.testing.assert_array_equal(sample["image"], expected)
    assert np.all(sample["mask"][:, :8] == 255)
