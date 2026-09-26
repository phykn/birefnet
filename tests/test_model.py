import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.model.swin as swin_model
from src.model import BiRefNet, Output
from src.model.decoder.net import Decoder
from src.model.swin import BasicLayer


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, 1)
        self.layers = nn.ModuleList([nn.Conv2d(4, 4, 1), nn.Conv2d(4, 8, 1), nn.Conv2d(8, 16, 1), nn.Conv2d(16, 32, 1)])
        for idx, channels in enumerate([4, 8, 16, 32]):
            setattr(self, f"norm{idx}", nn.BatchNorm2d(channels))

    def forward(self, x):
        x = self.patch_embed(F.avg_pool2d(x, 4))
        outs = []
        for idx, layer in enumerate(self.layers):
            if idx:
                x = F.avg_pool2d(x, 2)
            x = getattr(self, f"norm{idx}")(layer(x))
            outs.append(x)
        return outs


@pytest.fixture
def tiny_model(monkeypatch):
    monkeypatch.setattr("src.model.net.build_large", lambda **kwargs: TinyBackbone())
    return BiRefNet(channels=[32, 16, 8, 4])


def test_multiclass_forward_and_backward_cpu(tiny_model):
    x = torch.randn(1, 3, 64, 64)
    tiny_model.train()
    output = tiny_model(x)
    assert isinstance(output, Output)
    assert len(output.logits) == 4
    assert output.logits[-1].shape == (1, 4, 64, 64)
    assert all(pred.shape[1] == 4 for pred in output.logits)
    assert output.gdt is not None
    preds, labels = output.gdt
    assert len(preds) == len(labels) == 3
    assert all(pred.shape[1] == label.shape[1] == 1 for pred, label in zip(preds, labels))
    assert all(not label.requires_grad for label in labels)
    loss = sum(F.cross_entropy(pred, torch.zeros(pred.shape[0], *pred.shape[2:], dtype=torch.long)) for pred in output.logits)
    loss += sum(F.binary_cross_entropy_with_logits(pred, F.interpolate(label, size=pred.shape[2:])) for pred, label in zip(preds, labels))
    loss.backward()
    assert tiny_model.decoder.conv_out1[0].weight.grad is not None
    assert tiny_model.squeeze_module[0].conv_out.weight.grad is not None
    assert all(param.grad is None for param in tiny_model.bb.parameters())
    tiny_model.eval()
    with torch.inference_mode():
        output = tiny_model(x)
    assert isinstance(output, Output)
    assert output.gdt is None
    assert output.logits[-1].shape == (1, 4, 64, 64)


def test_only_decoder_and_squeeze_are_trainable(tiny_model):
    tiny_model.configure_finetune()
    assert all(not param.requires_grad for param in tiny_model.bb.parameters())
    for module in (tiny_model.squeeze_module, tiny_model.decoder):
        assert all(param.requires_grad for param in module.parameters())
    assert tiny_model.list_trainable()
    assert tiny_model.stats["total"] == tiny_model.stats["frozen"] + tiny_model.stats["trainable"]
    tiny_model.train()
    assert not tiny_model.bb.training
    assert all(not module.training for module in tiny_model.modules() if isinstance(module, nn.BatchNorm2d))


def test_batchnorm_stays_frozen_after_eval_train_cycle(tiny_model):
    tiny_model.eval()
    tiny_model.train()
    assert not tiny_model.bb.training
    assert all(not module.training for module in tiny_model.modules() if isinstance(module, nn.BatchNorm2d))


def test_guidance_marks_foreground_class_interfaces():
    decoder = Decoder([64, 32, 16, 8])
    logits = torch.full((1, 4, 8, 8), -8.0, requires_grad=True)
    with torch.no_grad():
        logits[:, 1, :, :4] = 8
        logits[:, 2, :, 4:] = 8
    label = decoder.guidance(logits, torch.ones(1, 1, 8, 8))
    assert label.shape == (1, 1, 8, 8)
    assert label.max() > 0
    assert label[..., 3].sum() > label[..., 0].sum()
    assert not label.requires_grad


def test_checkpointed_basic_layer_skips_checkpoint_without_grad(monkeypatch):
    layer = BasicLayer(
        dim=12,
        depth=1,
        num_heads=2,
        window_size=4,
        mlp_ratio=2.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=True,
    ).eval()

    def fail(*args, **kwargs):
        raise AssertionError("inference must not use gradient checkpointing")

    monkeypatch.setattr("src.model.swin.checkpoint.checkpoint", fail)
    with torch.inference_mode():
        output, *_ = layer(torch.randn(1, 64, 12), 8, 8)

    assert output.shape == (1, 64, 12)


def test_checkpointed_basic_layer_keeps_checkpoint_with_grad(monkeypatch):
    layer = BasicLayer(
        dim=12,
        depth=1,
        num_heads=2,
        window_size=4,
        mlp_ratio=2.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=True,
    ).eval()
    original = swin_model.checkpoint.checkpoint
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(swin_model.checkpoint, "checkpoint", tracked)
    layer(torch.randn(1, 64, 12), 8, 8)

    assert calls == [1]


def test_full_checkpoint_predictor_skips_missing_base_weights(tiny_model, tmp_path):
    from omegaconf import OmegaConf
    from src.build.model import build_predictor

    path = tmp_path / "model.pth"
    torch.save({
        "format": "birefnet-multiclass-v1", "model": tiny_model.state_dict(),
        "num_classes": 4, "preprocess": {"size": 1024},
        "teacher": {"legacy_weight": torch.ones(2)},
    }, path)
    cfg = OmegaConf.create({"birefnet": {
        "weight": "absent_base.pth", "channels": [32, 16, 8, 4],
        "grad_checkpoint": False, "num_classes": 4,
    }})
    loaded = build_predictor(cfg, str(path), torch.device("cpu"))
    assert loaded.training is False
    assert loaded.loaded_meta["preprocess"] == {"size": 1024}
    assert "teacher" not in loaded.loaded_meta
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, tiny_model.state_dict()[key])


def test_checkpointed_partial_layer_receives_gradient_with_frozen_input():
    layer = BasicLayer(
        dim=12, depth=1, num_heads=2, window_size=4, mlp_ratio=2.0,
        drop_path=0.0, downsample=None, use_checkpoint=True,
    )
    out, *_ = layer(torch.randn(1, 64, 12), 8, 8)
    out.square().mean().backward()
    assert all(param.grad is not None for param in layer.parameters())
