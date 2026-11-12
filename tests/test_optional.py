import pytest
import torch
from torch.nn import functional as F
from src.config import load_config, parse_config
from src.model import BiRefNet, Output
from src.train.loss import TrainLoss
from tests.model_utils import TinyBackbone
from tests.train_utils import make_loss, trainer


@pytest.mark.parametrize(
    "key,value",
    [
        ("model.depth", "yes"),
        ("loss.boundary", -1),
        ("loss.boundary_width", 0),
        ("loss.aux", -1),
        ("loss.label_blur", 11),
        ("loss.teacher", 1),
        ("loss.gce_q", 1.5),
        ("loss.teacher_start", 0),
    ],
)
def test_optional_config(key, value):
    section, field = key.split(".")
    with pytest.raises(ValueError, match=key):
        parse_config({section: {field: value}})


def test_optional_features_are_off_by_default():
    cfg = load_config()
    assert cfg.train.crop == 1024 and not cfg.model.depth
    assert all(
        getattr(cfg.loss, k) == 0 for k in ("boundary", "aux", "label_blur")
    )


def test_band_excludes_ignore_and_crop_edges():
    pred = torch.zeros(1, 4, 16, 16, requires_grad=True)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, 8:] = 255
    loss = make_loss(boundary=1)
    assert loss(Output([pred]), {"mask": target})[0]["boundary"] == 0
    uniform = torch.full_like(target, 2)
    assert loss(Output([pred]), {"mask": uniform})[0]["boundary"] == 0
    target[:, :, 8:] = 1
    assert loss(Output([pred]), {"mask": target})[0]["boundary"] > 0
    cut = torch.zeros(1, 1, 16, 16)
    cut[:, :, :, 7:9] = 1
    assert loss(Output([pred]), {"mask": target, "cut": cut})[0]["boundary"] == 0


def test_auxiliary_is_optional_and_backpropagates(monkeypatch):
    monkeypatch.setattr("src.model.net.build_large", TinyBackbone)
    m = BiRefNet(channels=[32, 16, 8, 4])
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
    _, loss = make_loss(aux=1)(
        aux, {"mask": torch.zeros(1, 64, 64, dtype=torch.long)}
    )
    loss.backward()
    assert m.decoder.gdt_convs_pred_2[0].weight.grad.abs().sum() > 0
    assert all(not t.requires_grad for t in aux.gdt[1])
    m.eval()
    assert m(x).gdt is None


class PreparedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Conv2d(3, 4, 1)
        self.num_classes = 4
        self.preparations = 0
        self.calls = 0

    def prepare(self, x):
        self.preparations += 1
        return x

    def forward(self, x, prepared=None):
        self.calls += 1
        if prepared is None:
            prepared = self.prepare(x)
        return Output([self.decoder(prepared)])

    def list_trainable(self):
        return list(self.parameters())


def test_teacher_ramp_and_shared_features(tmp_path):
    m = PreparedModel()
    t = trainer(tmp_path, m)
    t.cfg.loss.teacher = 0.3
    t.cfg.loss.teacher_start = 10
    shares = []
    for step in (0, 10, 15, 20, 100):
        t.global_step = step
        shares.append(t.teacher_share())
    assert shares == pytest.approx([0, 0, 0.15, 0.3, 0.3])
    with torch.no_grad():
        t.ema.params["decoder.weight"].zero_()
        t.ema.params["decoder.bias"].copy_(torch.tensor([0.0, 9.0, 0.0, 0.0]))
    x = torch.randn(1, 3, 16, 16)
    p = t.predict_teacher(x, m.prepare(x))
    assert (p.argmax(1) == 1).all()
    assert m.training and m.decoder.bias.abs().max() < 9
    m.calls = m.preparations = 0
    seen = []
    criterion = t.criterion

    def capture(out, batch, **kw):
        seen.append(kw)
        return criterion(out, batch, **kw)

    t.criterion = capture
    t.global_step = 20
    t.step()
    assert m.calls == 2 and m.preparations == 1
    assert seen[0]["teacher_weight"] == pytest.approx(0.3)
    assert not seen[0]["teacher"].requires_grad
    m.calls = m.preparations = 0
    t.cfg.loss.teacher = 0
    t.step()
    assert m.calls == 1 and m.preparations == 1


def test_v3_before_optional_fields_can_resume(tmp_path):
    t = trainer(tmp_path)
    t.step()
    t.save()
    path = tmp_path / "weights/last.train.pth"
    state = torch.load(path, weights_only=True)
    for key in ("depth", "depth_weight"):
        state["config"]["model"].pop(key)
    for key in (
        "boundary",
        "boundary_width",
        "aux",
        "label_blur",
        "teacher",
        "teacher_start",
    ):
        state["config"]["loss"].pop(key)
    torch.save(state, path)
    restored = trainer(tmp_path)
    restored.resume(path)
    assert restored.global_step == 1


def test_real_optional_step_and_roundtrip(tmp_path):
    from src.config import resolve
    from src.model.checkpoint import build, load
    from src.train.trainer import Trainer
    from src.train.schedule import cosine_schedule

    cfg = load_config()
    if (
        not resolve(cfg.model.weight).is_file()
        or not resolve(cfg.model.depth_weight).is_dir()
    ):
        pytest.skip("local weights unavailable")
    cfg.model.classes = 3
    cfg.model.depth = True
    cfg.loss.boundary = 0.5
    cfg.loss.aux = 1.0
    cfg.loss.label_blur = 1.0
    cfg.loss.teacher = 0.3
    cfg.loss.teacher_start = 1
    cfg.train.steps = 4
    cfg.train.warmup = 0
    cfg.train.accum = 1
    cfg.train.crop = 64
    model = build(cfg)
    opt = torch.optim.AdamW(model.list_trainable(), lr=cfg.train.lr)
    sch = cosine_schedule(opt, 4, cfg.train.min_lr)
    batch = {
        "image": torch.randn(1, 3, 64, 64),
        "mask": torch.randint(0, 3, (1, 64, 64)),
        "cut": torch.zeros(1, 1, 64, 64),
    }
    batch["mask"][:, :4] = 255
    criterion = TrainLoss(cfg)
    t = Trainer(model, [batch], [], criterion, opt, sch, tmp_path, cfg)
    t.global_step = 2
    values, updated = t.step()
    assert (
        updated
        and t.global_step == 3
        and all(torch.isfinite(torch.tensor(v)) for v in values.values())
    )
    assert values["aux"] > 0 and values["boundary"] > 0
    assert model.decoder.dpt_blks[0].conv_out.weight.grad.abs().sum() > 0
    assert model.decoder.gdt_convs_pred_2[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.depth.parameters())
    t.save()
    model.eval()
    with t.ema.applied(), torch.no_grad():
        expected = model(batch["image"]).logits[-1]
    del t, opt, sch, model
    restored = load(tmp_path / "weights/last.pth", torch.device("cpu"))
    with torch.no_grad():
        actual = restored(batch["image"]).logits[-1]
    torch.testing.assert_close(actual, expected)
