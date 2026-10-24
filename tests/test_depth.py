from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model import BiRefNet
from src.model.checkpoint import load_base
from src.model.depth import DEPTH_SIZE, Depth, rescale
from test_model import TinyBackbone

WEIGHT = Path(__file__).resolve().parents[1] / "weight/Depth-Anything-V2-Small-hf"


class FakeDepthNet(nn.Module):
    """Stands in for Depth Anything: brighter input reads as closer."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen = []

    def forward(self, pixel_values):
        self.seen.append(pixel_values.shape)
        return SimpleNamespace(predicted_depth=pixel_values.mean(dim=1) * self.scale)


@pytest.fixture
def depth_model(monkeypatch):
    monkeypatch.setattr("src.model.net.build_large", lambda **kwargs: TinyBackbone())
    return BiRefNet(channels=[32, 16, 8, 4], depth=Depth(FakeDepthNet()))


def test_rescale_is_per_image_and_keeps_padding_zero():
    depth = torch.stack([torch.linspace(0, 1, 100), torch.linspace(0, 1000, 100)]).reshape(2, 1, 10, 10)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[0, :, :, :5] = False
    out = rescale(depth, valid)
    assert torch.all(out[0][~valid[0]] == 0)
    for image, inside in zip(out, valid):
        values = image[inside]
        assert values.min() == 0 and values.max() == 1
    # The second image's larger scale does not squash the first one.
    assert out[0][valid[0]].std() > 0.2


def test_depth_ignores_padding_and_runs_at_its_own_resolution():
    net = FakeDepthNet()
    depth = Depth(net)
    x = torch.randn(2, 3, 64, 48)
    x[:, :, :, :8] = 0
    out = depth(x)
    assert net.seen == [(2, 3, DEPTH_SIZE, DEPTH_SIZE)]
    assert out.shape == (2, 1, 64, 48)
    assert torch.all(out[..., :8] == 0)
    assert out.min() >= 0 and out.max() <= 1
    assert not out.requires_grad


def test_depth_estimator_stays_frozen_in_training(depth_model):
    depth_model.train()
    assert not depth_model.depth.training and not depth_model.depth.net.training
    assert all(not param.requires_grad for param in depth_model.depth.parameters())
    assert all(param.requires_grad for param in depth_model.decoder.dpt_blks.parameters())


def test_depth_branch_starts_as_the_plain_decoder(depth_model, monkeypatch):
    monkeypatch.setattr("src.model.net.build_large", lambda **kwargs: TinyBackbone())
    plain = BiRefNet(channels=[32, 16, 8, 4])
    plain.load_state_dict({
        key: value for key, value in depth_model.state_dict().items()
        if not key.startswith(("depth.", "decoder.dpt_blks."))
    })
    x = torch.randn(1, 3, 64, 64)
    depth_model.eval()
    plain.eval()
    with torch.no_grad():
        torch.testing.assert_close(depth_model(x).logits[-1], plain(x).logits[-1])


def test_depth_branch_learns(depth_model):
    depth_model.train()
    output = depth_model(torch.randn(1, 3, 64, 64))
    sum(logit.square().mean() for logit in output.logits).backward()
    assert all(block.conv_out.weight.grad.abs().sum() > 0 for block in depth_model.decoder.dpt_blks)
    assert depth_model.depth.net.scale.grad is None


def test_base_weights_load_into_a_depth_model(depth_model, monkeypatch):
    monkeypatch.setattr("src.model.net.build_large", lambda **kwargs: TinyBackbone())
    base = BiRefNet(channels=[32, 16, 8, 4]).state_dict()
    branch = depth_model.decoder.dpt_blks[0].conv_out.weight.detach().clone()
    load_base(depth_model, base)
    torch.testing.assert_close(depth_model.squeeze_module[0].conv_out.weight, base["squeeze_module.0.conv_out.weight"])
    torch.testing.assert_close(depth_model.decoder.dpt_blks[0].conv_out.weight, branch)


@pytest.mark.skipif(not WEIGHT.is_dir(), reason="Depth Anything weights are not downloaded")
def test_real_depth_anything_loads_offline():
    depth = Depth.load(WEIGHT)
    out = depth(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
    assert 0 <= out.min() and out.max() <= 1


@pytest.mark.parametrize("enabled", [True, False])
def test_depth_switch_decides_whether_the_estimator_is_loaded(monkeypatch, enabled):
    from omegaconf import OmegaConf
    from src.build.model import build

    loaded = []
    monkeypatch.setattr("src.model.net.build_large", lambda **kwargs: TinyBackbone())
    monkeypatch.setattr(Depth, "load", classmethod(lambda cls, path: loaded.append(path) or cls(FakeDepthNet())))
    cfg = OmegaConf.create({"model": {"channels": [32, 16, 8, 4], "classes": 4,
                                      "depth": enabled, "depth_weight": "depth_folder"}})
    model = build(cfg, load_pretrained=False)
    assert (model.depth is not None) is enabled
    assert loaded == (["depth_folder"] if enabled else [])


@pytest.mark.parametrize("value", ["weight/Depth-Anything-V2-Small-hf", None, 1])
def test_depth_switch_must_be_a_bool(value):
    from omegaconf import OmegaConf
    from src.build.model import build

    with pytest.raises(ValueError, match="model.depth must be a bool"):
        build(OmegaConf.create({"model": {"classes": 4, "depth": value}}), load_pretrained=False)
