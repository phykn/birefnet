from contextlib import contextmanager

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.output import Output
from src.predict.inference import predict_logits, probabilities
from src.predict.tile import Tile, plan, weigh


@pytest.mark.parametrize("support", [(True, False), (False, True)])
def test_autocast_uses_model_device_and_restores_current_device(monkeypatch, support):
    from src.device import autocast, autocast_dtype

    current = [0]
    queried = []

    @contextmanager
    def device(target):
        previous = current[0]
        current[0] = target.index
        try:
            yield
        finally:
            current[0] = previous

    def supported(including_emulation):
        assert including_emulation is False
        queried.append(current[0])
        return support[current[0]]

    def make_autocast(device_type, dtype, enabled=True):
        assert device_type == "cuda" and enabled
        assert current == [1]
        return dtype

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    monkeypatch.setattr(torch.amp, "autocast", make_autocast)

    target = torch.device("cuda:1")
    dtype = autocast(target, dtype=autocast_dtype(target))

    assert dtype == (torch.bfloat16 if support[1] else torch.float16)
    assert queried == [1]
    assert current == [0]


@pytest.mark.parametrize(("device", "expected"), [("cuda", True), ("mps", True), ("cpu", False)])
def test_mixed_precision_runs_on_gpus_only(device, expected):
    from src.device import amp_enabled

    assert amp_enabled(torch.device(device)) is expected


class _ConstantModel(nn.Module):
    def __init__(self, logit: float = 0.0):
        super().__init__()
        self.num_classes = 4
        self.logit = nn.Parameter(torch.tensor([0., -1., -2., logit]))

    def forward(self, x):
        output = self.logit[None, :, None, None].expand(x.shape[0], 4, x.shape[2], x.shape[3])
        return Output(logits=[output])


class _MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        value = x.mean(dim=(1, 2, 3), keepdim=True) + self.anchor
        output = value.expand(x.shape[0], 4, x.shape[2], x.shape[3])
        return Output(logits=[output])


def test_cpu_inference_does_not_query_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CPU inference must not access CUDA")

    monkeypatch.setattr(torch.cuda, "device", fail)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", fail)
    predict_logits(_ConstantModel().eval(), np.zeros((8, 8, 3), np.uint8), size=16)


def test_tile_planner_covers_image_and_aligns_last_tile():
    boxes = plan(45, 61, grid=4, overlap=1 / 3)
    assert max(box.bottom for box in boxes) == 45
    assert max(box.right for box in boxes) == 61
    coverage = np.zeros((45, 61), dtype=np.int32)
    for box in boxes:
        coverage[box.top : box.bottom, box.left : box.right] += 1
    assert np.all(coverage > 0)


def test_tile_planner_uses_requested_grid():
    for grid in (1, 2, 3, 4):
        assert len(plan(1500, 1500, grid=grid)) == grid**2


def test_tile_planner_keeps_overlap_at_least_one_third():
    for height, width, grid in [
        (1500, 1500, 2),
        (2000, 2000, 3),
        (3000, 4000, 4),
    ]:
        boxes = plan(height, width, grid=grid, overlap=1 / 3)
        for box in boxes:
            if box.overlap_top:
                assert box.overlap_top / box.height >= 1 / 3
            if box.overlap_bottom:
                assert box.overlap_bottom / box.height >= 1 / 3
            if box.overlap_left:
                assert box.overlap_left / box.width >= 1 / 3
            if box.overlap_right:
                assert box.overlap_right / box.width >= 1 / 3

    with pytest.raises(ValueError, match="overlap"):
        plan(100, 100, grid=2, overlap=0.2)


def test_blend_window_is_symmetric_and_keeps_outer_edges_nonzero():
    box = Tile(
        0,
        0,
        12,
        16,
        overlap_top=4,
        overlap_left=5,
        overlap_bottom=4,
        overlap_right=5,
    )
    window = weigh(box)
    np.testing.assert_allclose(window, window[::-1, :])
    np.testing.assert_allclose(window, window[:, ::-1])

    outer = weigh(Tile(0, 0, 12, 16, overlap_bottom=4, overlap_right=5))
    assert np.all(outer[0] > 0)
    assert np.all(outer[:, 0] > 0)


def test_neighboring_cosine_windows_are_complementary():
    left, right = plan(1500, 1500, grid=2)[:2]
    overlap = left.overlap_right
    assert overlap == right.overlap_left
    left_window = weigh(left)
    right_window = weigh(right)
    np.testing.assert_allclose(
        left_window[0, -overlap:] + right_window[0, :overlap],
        1.0,
        atol=1e-6,
    )


def test_one_by_one_skips_tile_planner(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("tile planner should not run")

    monkeypatch.setattr("src.predict.inference.plan", fail)
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((45, 61, 3), dtype=np.uint8)
    logits = predict_logits(model, image, size=32)

    assert logits.shape == (4, *image.shape[:2])
    np.testing.assert_allclose(logits[3], 1.25, atol=1e-6)


def test_tile_batch_does_not_change_logits():
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((45, 61, 3), dtype=np.uint8)
    first = predict_logits(
        model,
        image,
        size=32,
        tiles=[3],
        overlap=1 / 3,
        tile_batch=1,
    )
    second = predict_logits(
        model,
        image,
        size=32,
        tiles=[3],
        overlap=1 / 3,
        tile_batch=3,
    )
    assert first.shape == (4, *image.shape[:2])
    np.testing.assert_allclose(first[3], 1.25, atol=1e-6)
    np.testing.assert_allclose(first, second, atol=1e-6)


@pytest.mark.parametrize("tiles", [(1,), (3,), (1, 3)])
def test_predictions_do_not_share_storage_with_model_or_input(tiles):
    model = _ConstantModel(logit=1.25).eval()
    weights = model.logit.detach().clone()
    image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    original = image.copy()

    logits = predict_logits(model, image, size=32, tiles=tiles)
    logits.fill(0)

    torch.testing.assert_close(model.logit, weights)
    np.testing.assert_array_equal(image, original)
    repeated = predict_logits(model, image, size=32, tiles=tiles)
    np.testing.assert_allclose(repeated[3], 1.25, atol=1e-6)


def test_multiple_grids_are_streamed_and_averaged_at_logit_level(monkeypatch):
    model = _MeanModel().eval()
    image = np.zeros((45, 61, 3), dtype=np.uint8)
    image[:, 31:] = 255
    single = predict_logits(model, image, size=32, tiles=[1])
    tiled = predict_logits(model, image, size=32, tiles=[2])

    def fail(*args, **kwargs):
        raise AssertionError("grid outputs must not be collected for np.mean")

    with monkeypatch.context() as patch:
        patch.setattr("src.predict.inference.np.mean", fail)
        combined = predict_logits(model, image, size=32, tiles=[1, 2])

    np.testing.assert_allclose(combined, (single + tiled) / 2, atol=1e-6)


def test_tiles_accept_any_positive_grid():
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((17, 23, 3), dtype=np.uint8)
    logits = predict_logits(model, image, size=32, tiles=[4], tile_batch=5)
    np.testing.assert_allclose(logits[3], 1.25, atol=1e-6)

    with pytest.raises(ValueError, match="positive integers"):
        predict_logits(model, image, size=32, tiles=[0])


def test_labels_and_softmax_preserve_all_classes():
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((17, 43, 3), dtype=np.uint8)
    logits = predict_logits(model, image, size=32, tiles=[1, 3])
    assert logits.shape == (4, 17, 43)
    assert np.all(logits.argmax(axis=0) == 3)
    probs = probabilities(logits)
    np.testing.assert_allclose(probs.sum(axis=0), 1., atol=1e-6)
    assert np.all(probs[3] > probs[0])


class _SpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        output = torch.cat([x[:, :1], -x[:, :1], x[:, 1:2], -x[:, 1:2]], dim=1)
        return Output(logits=[output + self.anchor])


@pytest.mark.parametrize("tiles", [(1,), (2,), (1, 3)])
def test_non_square_orientation_survives_restore_and_tiling(tiles):
    image = np.zeros((31, 65, 3), dtype=np.uint8)
    image[:, :32, 0] = 255
    image[:, 32:, 1] = 255
    logits = predict_logits(_SpatialModel(), image, size=64, tiles=tiles)
    assert logits.shape == (4, 31, 65)
    assert np.all(logits.argmax(axis=0)[4:-4, 4:25] == 0)
    assert np.all(logits.argmax(axis=0)[4:-4, 40:-4] == 2)


@pytest.mark.parametrize("grid", [1, 2])
def test_sem_inference_generates_features_from_each_input_patch(monkeypatch, grid):
    from src.prepare.fit import fit_tensor

    image = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    seen = []

    def capture(model, tensors, device):
        seen.extend(tensors)
        return np.zeros((len(tensors), 4, 32, 32), np.float32)

    monkeypatch.setattr("src.predict.inference._infer", capture)
    predict_logits(_ConstantModel(), image, size=32, is_sem=True, tiles=(grid,))
    boxes = plan(32, 48, grid=grid)
    assert len(seen) == len(boxes)
    for actual, box in zip(seen, boxes):
        patch = image[box.top:box.bottom, box.left:box.right]
        expected, _ = fit_tensor(patch, size=32, is_sem=True)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("explicit", [False, True])
def test_prediction_config_uses_saved_config_unless_explicit(tmp_path, explicit):
    from omegaconf import OmegaConf
    from src.predict.model import load_run_config

    weights = tmp_path / "weights"
    weights.mkdir()
    OmegaConf.save({"model": {"classes": 5}}, tmp_path / "config.yaml")
    config = tmp_path / "custom.yaml"
    OmegaConf.save({"model": {"classes": 6}}, config)
    cfg = load_run_config(weights / "last.pth", config if explicit else None)
    assert cfg.model.classes == (6 if explicit else 5)


def test_prediction_config_falls_back_to_model_defaults(tmp_path):
    from src.predict.model import load_run_config

    cfg = load_run_config(tmp_path / "weights/last.pth")
    assert cfg.model.classes == 4
