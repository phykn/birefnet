import numpy as np
import pytest

from src.data.augment import rotate_flip


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("horizontal", range(2))
def test_rotate_flip_applies_same_transform(
    monkeypatch, rotation, horizontal
):
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8, 1)
    cut = np.eye(8, dtype=np.uint8)
    values = iter((rotation, horizontal))
    monkeypatch.setattr(np.random, "randint", lambda *_: next(values))
    img2, msk2, cut2 = rotate_flip(image, mask, cut)
    expected_img = np.rot90(image, k=rotation, axes=(0, 1))
    expected_mask = np.rot90(mask, k=rotation, axes=(0, 1))
    expected_cut = np.rot90(cut, k=rotation, axes=(0, 1))
    if horizontal:
        expected_img = np.flip(expected_img, axis=1)
        expected_mask = np.flip(expected_mask, axis=1)
        expected_cut = np.flip(expected_cut, axis=1)
    assert np.array_equal(img2, np.ascontiguousarray(expected_img))
    assert np.array_equal(msk2, np.ascontiguousarray(expected_mask))
    assert np.array_equal(cut2, np.ascontiguousarray(expected_cut))
    assert all(x.flags["C_CONTIGUOUS"] for x in (img2, msk2, cut2))
