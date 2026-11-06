import numpy as np
import pytest
from src.data.augment import jitter, gamma, illumination, blur, noise


@pytest.mark.parametrize(
    "bright,contrast,expected",
    [(0.2, 0.0, 115), (-0.2, 0.0, 13), (0.0, 0.4, 39), (0.0, -0.4, 89)],
)
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
