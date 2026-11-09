import numpy as np
import pytest
import torch

from src.predict import plan, predict_logits, weigh
from tests.train_utils import Tiny


@pytest.mark.parametrize("shape", [(1, 3), (17, 29), (64, 64)])
@pytest.mark.parametrize("overlap", [1 / 3, 0.8])
def test_tiles_cover_every_pixel_with_positive_weight(shape, overlap):
    height, width = shape
    weight = np.zeros(shape, dtype=np.float32)
    for tile in plan(height, width, grid=4, overlap=overlap):
        assert 0 <= tile.top < tile.bottom <= height
        assert 0 <= tile.left < tile.right <= width
        blend = weigh(tile)
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
    logits = predict_logits(model, image, size=64, tiles=tiles, tile_batch=2)
    expected = np.broadcast_to(np.array([-2, 0, 1, 3])[:, None, None], (4, 17, 29))
    np.testing.assert_allclose(logits, expected, atol=1e-6, rtol=0)
