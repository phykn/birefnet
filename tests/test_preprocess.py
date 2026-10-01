import numpy as np
import pytest

from src.prepare.convert import MEAN, STD, _enhance, convert, normalize
from src.prepare.fit import fit_image, fit_mask, fit_tensor, plan, restore
from src.prepare.spec import PreprocessSpec


def test_normalize_known_rgb_pixel():
    image = np.array([[[255, 0, 127]]], dtype=np.uint8)
    expected = (image.astype(np.float32) / 255.0 - MEAN) / STD
    np.testing.assert_allclose(normalize(image), expected)


def test_modes_preserve_declared_channel_semantics():
    image = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(convert(image, False), image)
    assert convert(image, True).shape == image.shape


def test_fixed_canvas_and_valid_mask_for_wide_image():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    tensor, valid, fit = fit_image(image, size=32)
    assert tensor.shape == (3, 32, 32)
    assert valid.shape == (1, 32, 32)
    assert fit.dst_h == 16
    assert fit.dst_w == 32
    assert valid.sum() == 16 * 32
    assert np.all(valid[:, fit.top : fit.top + 16] == 1)


def test_mask_resize_stays_binary_and_matches_valid_geometry():
    mask = np.zeros((15, 30), dtype=np.float32)
    mask[:, 10:] = 1
    fit = plan(15, 30, size=32)
    prepared = fit_mask(mask, fit)
    assert set(np.unique(prepared)) <= {0.0, 1.0}
    assert prepared.shape == (1, 32, 32)


def test_mask_fit_rejects_source_geometry_mismatch():
    fit = plan(15, 30, size=32)
    with pytest.raises(ValueError, match="source shape"):
        fit_mask(np.zeros((15, 20), dtype=np.int64), fit)


def test_restore_logit_returns_original_shape_and_removes_padding():
    image = np.zeros((17, 43, 3), dtype=np.uint8)
    _, _, fit = fit_image(image, size=32)
    canvas = np.zeros((32, 32), dtype=np.float32)
    y0, x0 = fit.top, fit.left
    canvas[y0 : y0 + fit.dst_h, x0 : x0 + fit.dst_w] = 3.0
    restored = restore(canvas, fit)
    assert restored.shape == (17, 43)
    np.testing.assert_allclose(restored, 3.0)


def test_fit_tensor_matches_fit_image_without_allocating_valid_contract():
    image = np.arange(15 * 30 * 3, dtype=np.uint8).reshape(15, 30, 3)
    expected, _, expected_fit = fit_image(image, size=32)
    actual, actual_fit = fit_tensor(image, size=32)

    np.testing.assert_array_equal(actual, expected)
    assert actual_fit == expected_fit


def test_preprocess_spec_round_trip_and_legacy_default():
    spec = PreprocessSpec(is_sem=True)
    assert PreprocessSpec.from_meta({"preprocess": spec.to_meta()}) == spec
    assert PreprocessSpec.from_meta({}) == PreprocessSpec()

    with np.testing.assert_raises(RuntimeError):
        PreprocessSpec.from_meta({"preprocess": {"size": 640.5, "is_sem": False}})


def test_sem_channels():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    features = convert(gray, True)
    low, high = np.percentile(gray, (1, 99))
    expected = np.rint(np.clip((gray.astype(float) - low) / (high - low), 0, 1) * 255)
    np.testing.assert_array_equal(features[..., 0], gray)
    np.testing.assert_array_equal(features[..., 1], expected)
    np.testing.assert_array_equal(features[..., 2], _enhance(gray))
    spec = PreprocessSpec(is_sem=True)
    assert PreprocessSpec.from_meta({"preprocess": spec.to_meta()}) == spec


def test_constant_sem_image_is_finite():
    for value in (0, 127, 255):
        gray = np.full((16, 16), value, np.uint8)
        features = convert(gray, True)
        assert not features[..., 1].any()
        assert np.isfinite(normalize(features)).all()


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_sem_flag_rejects_non_booleans(value):
    with pytest.raises(ValueError, match="bool"):
        convert(np.zeros((8, 8), np.uint8), value)
    with pytest.raises(ValueError, match="bool"):
        PreprocessSpec(is_sem=value)


@pytest.mark.parametrize("mode, expected", [("rgb", False), ("sem_features", True)])
def test_legacy_preprocess_metadata_maps_only_equivalent_inputs(mode, expected):
    spec = PreprocessSpec.from_meta({"preprocess": {"size": 1024, "mode": mode}})
    assert spec.is_sem is expected
    assert spec.to_meta() == {"size": 1024, "is_sem": expected}


@pytest.mark.parametrize("mode", ["gray_repeat", "gray_features"])
def test_removed_preprocess_modes_fail_explicitly(mode):
    with pytest.raises(RuntimeError, match="preprocess"):
        PreprocessSpec.from_meta({"preprocess": {"size": 1024, "mode": mode}})


@pytest.mark.parametrize("size", [512, 2048])
def test_checkpoint_input_size_must_match_pretrained_resolution(size):
    with pytest.raises(RuntimeError, match="preprocess"):
        PreprocessSpec.from_meta({"preprocess": {"size": size, "is_sem": True}})


def test_input_size_cannot_be_overridden():
    assert PreprocessSpec().size == 1024
    with pytest.raises(TypeError):
        PreprocessSpec(size=512)
