import numpy as np
import pytest
import torch

from src.evaluate import compute_scores, count_confusion
from src.predict import plan_tiles, predict, weigh_tile
from tests.helpers import Tiny


@pytest.mark.parametrize("shape", [(1, 3), (17, 29), (64, 64)])
@pytest.mark.parametrize("overlap", [1 / 3, 0.8])
def test_tiles_cover_every_pixel_with_positive_weight(shape, overlap):
    height, width = shape
    weight = np.zeros(shape, dtype=np.float32)
    for tile in plan_tiles(height, width, 4, overlap):
        assert 0 <= tile.top < tile.bottom <= height
        assert 0 <= tile.left < tile.right <= width
        blend = weigh_tile(tile)
        assert np.isfinite(blend).all() and (blend > 0).all()
        weight[tile.top : tile.bottom, tile.left : tile.right] += blend
    assert (weight > 0).all()


@pytest.mark.parametrize("tiles", [(1,), (3,), (1, 3)])
def test_tile_blending_preserves_constant_logits(tiles):
    model = Tiny()
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.bias.copy_(torch.tensor([-2.0, 0.0, 1.0, 3.0]))
    image = np.arange(17 * 29, dtype=np.uint8).reshape(17, 29)
    result = predict(model, image, size=64, tiles=tiles, tile_batch=2)
    expected = np.broadcast_to(np.array([-2, 0, 1, 3])[:, None, None], (4, 17, 29))
    np.testing.assert_allclose(result["logits"], expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(result["probs"].sum(axis=0), 1, atol=1e-6)
    assert (result["classes"] == 3).all()


def test_tiles_shape_and_batch():
    m = Tiny().eval()
    x = np.arange(35 * 47, dtype=np.uint8).reshape(35, 47)
    for tiles in ((1,), (1, 3)):
        a = predict(m, x, size=64, tiles=tiles, tile_batch=1)["logits"]
        b = predict(m, x, size=64, tiles=tiles, tile_batch=3)["logits"]
        assert a.shape == (4, 35, 47)
        np.testing.assert_allclose(a, b, atol=1.0e-6, rtol=0)


def test_metrics():
    s = compute_scores(torch.tensor([[4, 1], [2, 3]]))
    assert s["class_0_iou"] == 4 / 7 and s["class_1_iou"] == 0.5
    assert s["miou"] == (4 / 7 + 0.5) / 2 and s["pixel_accuracy"] == 0.7
    assert count_confusion(torch.tensor([0, 1]), torch.tensor([0, 255]), 2).sum() == 1
