import pytest
import torch
from torchvision.ops import deform_conv2d

from src.config import load_config, resolve
from src.model import BiRefNet
from src.model.checkpoint import (
    SEGMENTATION_HEADS,
    build_model,
    load_base,
    load_model,
    load_trained,
    select_trained,
)
from src.model.deform import DeformableConv2d, grid_deform_conv2d
from src.train.loss import TrainLoss
from src.train.schedule import cosine_schedule
from src.train.trainer import Trainer
from tests.helpers import TinyBackbone, make_loss

needs_weight = pytest.mark.skipif(
    not resolve(load_config().model.weight).is_file(), reason="base weight unavailable"
)


@needs_weight
def test_base_and_trained_keys():
    cfg = load_config()
    m = build_model(cfg)
    base = torch.load(resolve(cfg.model.weight), weights_only=True, map_location="cpu")
    current = m.state_dict()
    replaced = []
    for key, value in base.items():
        if key.rsplit(".", 1)[0] in SEGMENTATION_HEADS:
            assert value.shape[0] == 1 and current[key].shape[0] == cfg.model.classes
            replaced.append(key)
        else:
            torch.testing.assert_close(current[key], value, atol=0, rtol=0)
    assert len(replaced) == 8
    key = "bb.patch_embed.proj.weight"
    broken = dict(base)
    broken.pop(key)
    with pytest.raises(RuntimeError, match=key):
        load_base(m, broken)
    broken = dict(base)
    broken[key] = base[key][:1]
    with pytest.raises(RuntimeError, match=key):
        load_base(m, broken)
    trained = select_trained(current)
    trained.pop(next(iter(trained)))
    with pytest.raises(RuntimeError):
        load_trained(m, trained)


@needs_weight
def test_real_optional_step_and_roundtrip(tmp_path):
    cfg = load_config()
    cfg.model.classes = 3
    cfg.loss.boundary = 0.5
    cfg.loss.aux = 1.0
    cfg.loss.label_blur = 1.0
    cfg.loss.teacher = 0.3
    cfg.loss.teacher_start = 1
    cfg.train.steps = 4
    cfg.train.warmup = 0
    cfg.train.accum = 1
    cfg.train.crop = 64
    model = build_model(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.train.lr)
    sch = cosine_schedule(opt, 4, cfg.train.min_lr)
    batch = {
        "image": torch.randn(1, 3, 64, 64),
        "mask": torch.randint(0, 3, (1, 64, 64)),
        "cut": torch.zeros(1, 1, 64, 64),
    }
    batch["mask"][:, :4] = 255
    t = Trainer(model, [batch], [], TrainLoss(cfg), opt, sch, tmp_path, cfg)
    t.global_step = 2
    values, updated = t.step()
    assert updated and t.global_step == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in values.values())
    assert values["aux"] > 0 and values["boundary"] > 0
    assert model.decoder.gdt_convs_pred_2[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.bb.parameters())
    t.save()
    model.eval()
    with t.ema.applied(), torch.no_grad():
        expected = model(batch["image"]).logits[-1]
    del t, opt, sch, model
    restored = load_model(tmp_path / "weights/last.pth", torch.device("cpu"))
    with torch.no_grad():
        actual = restored(batch["image"]).logits[-1]
    torch.testing.assert_close(actual, expected)


def test_auxiliary_is_optional_and_backpropagates(monkeypatch):
    monkeypatch.setattr("src.model.net.build_large", TinyBackbone)
    m = BiRefNet(4, channels=[32, 16, 8, 4])
    x = torch.randn(1, 3, 64, 64)
    torch.manual_seed(19)
    plain = m(x)
    assert plain.gdt is None
    m.auxiliary = True
    torch.manual_seed(19)
    aux = m(x)
    assert aux.gdt is not None
    for a, b in zip(plain.logits, aux.logits):
        torch.testing.assert_close(a, b)
    _, loss = make_loss(aux=1)(aux, {"mask": torch.zeros(1, 64, 64, dtype=torch.long)})
    loss.backward()
    assert m.decoder.gdt_convs_pred_2[0].weight.grad.abs().sum() > 0
    assert all(not t.requires_grad for t in aux.gdt[1])
    m.eval()
    assert m(x).gdt is None


@pytest.mark.parametrize("kernel", [1, 3, 7])
def test_grid_deform_conv_matches_torchvision(kernel):
    padding = kernel // 2
    torch.manual_seed(0)
    out_h, out_w = 9, 11
    inputs = (
        torch.randn(2, 4, 9, 11, dtype=torch.float64),
        # Large offsets also sample outside the image.
        torch.randn(2, 2 * kernel * kernel, out_h, out_w, dtype=torch.float64) * 3,
        torch.randn(5, 4, kernel, kernel, dtype=torch.float64),
        torch.rand(2, kernel * kernel, out_h, out_w, dtype=torch.float64),
    )
    for value in inputs:
        value.requires_grad_(True)
    x, offset, weight, mask = inputs
    expected = deform_conv2d(x, offset, weight, padding=padding, mask=mask)
    actual = grid_deform_conv2d(x, offset, weight, padding=padding, mask=mask)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs),
    ):
        torch.testing.assert_close(got, want)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_deformable_conv_trains_on_mps_without_cpu_fallback():
    torch.manual_seed(0)
    layer = DeformableConv2d(4, 5, 3)
    with torch.no_grad():
        layer.offset_conv.weight.normal_(0, 0.1)
    x = torch.randn(1, 4, 16, 16)
    expected = layer(x)
    layer.to("mps")
    actual = layer(x.to("mps"))
    actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    assert layer.offset_conv.weight.grad.abs().sum() > 0
