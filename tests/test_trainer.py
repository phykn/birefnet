import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.model.output import Output
from src.prepare.spec import PreprocessSpec
from src.predict.inference import predict_logits
from src.train.trainer import Trainer
from src.train.schedule import cosine_schedule


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.decoder = nn.Conv2d(3, 4, 1)
        self.calls = 0

    def prepare(self, x):
        return x

    def forward(self, x, prepared=None):
        self.calls += 1
        pred = self.decoder(x)
        if self.training:
            return Output(logits=[pred, pred])
        return Output(logits=[pred])

    def list_trainable(self):
        return list(self.parameters())



class _DummyCriterion(nn.Module):
    def forward(self, out, batch, **kwargs):
        if len(out.logits) > 1:
            logits = out.logits
            seg = sum(logit.mean() ** 2 for logit in logits) / len(logits)
            cons = (logits[0] - logits[-1]).pow(2).mean()
            aux = seg.new_zeros(())
            loss = seg + cons + aux
            return {"loss": loss, "seg": seg, "cons": cons, "aux": aux}, loss
        pred = out.logits[-1]
        seg = pred.mean() ** 2
        return {"loss": seg, "seg": seg}, seg


class _DummyDataset(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 8, 8),
            "mask": torch.randint(0, 4, (8, 8)),
            "valid": torch.ones(1, 8, 8),
        }


def _make_trainer(tmp_path, accum_steps=1, preprocess=None, max_grad_norm=1.0, **teacher):
    model = _DummyModel()
    train_loader = DataLoader(_DummyDataset(4), batch_size=2)
    criterion = _DummyCriterion()
    optimizer = torch.optim.AdamW(model.list_trainable(), lr=1e-2)
    scheduler = cosine_schedule(optimizer, steps=10, min_lr=1e-4, warmup=2)
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_pairs=[("image.png", "mask.png")],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=str(tmp_path),
        predictor=predict_logits,
        max_grad_norm=max_grad_norm,
        accum_steps=accum_steps,
        preprocess=preprocess,
        **teacher,
    )


def test_trainer_step_returns_loss_dict_and_updates_params(tmp_path):
    trainer = _make_trainer(tmp_path)
    before = trainer.model.decoder.weight.detach().clone()

    losses, updated = trainer.step()

    assert updated
    assert {"loss", "seg", "cons", "aux"} <= set(losses.keys())
    assert all(isinstance(v, float) for v in losses.values())
    after = trainer.model.decoder.weight
    assert not torch.allclose(before, after)


def test_trainer_step_with_accum(tmp_path):
    trainer = _make_trainer(tmp_path, accum_steps=2)
    losses, updated = trainer.step()
    assert updated
    assert "loss" in losses


def test_validation_and_best_checkpoint_use_ema_weights(tmp_path):
    trainer = _make_trainer(tmp_path)
    live = trainer.model.decoder.weight.detach().clone()
    trainer.ema.params["decoder.weight"].fill_(0.5)
    seen = []
    trainer.validator.validate = lambda: seen.append(trainer.model.decoder.weight.detach().clone()) or {"miou": 0.5}
    trainer._writer = SummaryWriter(log_dir=str(tmp_path / "logs"))
    try:
        trainer._evaluate()
    finally:
        trainer._writer.close()
    assert torch.all(seen[0] == 0.5)
    assert torch.equal(trainer.model.decoder.weight, live)
    best = torch.load(tmp_path / "weights" / "best_miou.pth", weights_only=True)
    assert torch.all(best["model"]["decoder.weight"] == 0.5)


def test_trainer_get_batch_wraps_around(tmp_path):
    trainer = _make_trainer(tmp_path)
    for _ in range(6):
        batch = trainer.next_batch()
        assert "image" in batch


def test_trainer_save_writes_full_model_and_resume_state(tmp_path):
    trainer = _make_trainer(
        tmp_path,
        preprocess=PreprocessSpec(is_sem=True),
    )
    trainer.save()
    weights_dir = os.path.join(trainer.save_dir, "weights")
    overlay_path = os.path.join(weights_dir, "last.pth")
    assert os.path.exists(overlay_path)
    assert os.path.exists(os.path.join(weights_dir, "last.train.pth"))
    overlay = torch.load(overlay_path, map_location="cpu", weights_only=True)
    assert overlay["preprocess"] == {
        "size": 1024,
        "is_sem": True,
    }


def test_trainer_resume_restores_step_model_optimizer_and_scheduler(tmp_path):
    preprocess = PreprocessSpec(is_sem=True)
    trainer = _make_trainer(tmp_path, preprocess=preprocess)
    _, updated = trainer.step()
    assert updated
    expected_weight = trainer.model.decoder.weight.detach().clone()
    expected_ema = trainer.ema.state_dict()
    expected_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.best_region = 0.6
    trainer.save()

    resumed = _make_trainer(tmp_path, preprocess=preprocess)
    resumed.load_resume(os.path.join(tmp_path, "weights", "last.train.pth"))
    assert resumed.global_step == trainer.global_step
    assert resumed.best_region == trainer.best_region
    assert torch.allclose(resumed.model.decoder.weight, expected_weight)
    assert resumed.optimizer.param_groups[0]["lr"] == expected_lr
    assert resumed.scheduler.last_epoch == trainer.scheduler.last_epoch
    for name, value in expected_ema.items():
        assert torch.allclose(resumed.ema.state_dict()[name], value)


@pytest.mark.parametrize(
    "preprocess",
    [PreprocessSpec(is_sem=True)],
)
def test_resume_rejects_preprocess_mismatch_before_loading_weights(tmp_path, preprocess):
    trainer = _make_trainer(tmp_path)
    trainer.step()
    trainer.save()
    resumed = _make_trainer(tmp_path, preprocess=preprocess)
    before = {name: value.clone() for name, value in resumed.model.state_dict().items()}

    with pytest.raises(RuntimeError, match="preprocess"):
        resumed.load_resume(str(tmp_path / "weights" / "last.train.pth"))

    assert resumed.global_step == 0
    assert not resumed.optimizer.state
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, before[name])


class _SkipScaler:
    def __init__(self):
        self.value = 2.0
        self.attempts = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.attempts += 1
        if self.attempts > 1:
            optimizer.step()

    def update(self):
        if self.attempts == 1:
            self.value = 1.0

    def get_scale(self):
        return self.value


class _AlwaysSkipScaler(_SkipScaler):
    def step(self, optimizer):
        self.attempts += 1

    def update(self):
        self.value /= 2


def test_overflow_skip_keeps_optimizer_dependent_state(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.scaler = _SkipScaler()
    weight = trainer.model.decoder.weight.detach().clone()
    ema = trainer.ema.state_dict()
    cycle = trainer.scheduler.last_epoch

    _, updated = trainer.step()

    assert not updated
    assert trainer.global_step == 0
    assert trainer.scheduler.last_epoch == cycle
    assert torch.allclose(trainer.model.decoder.weight, weight)
    for name, value in ema.items():
        assert torch.allclose(trainer.ema.state_dict()[name], value)


def test_training_retries_skipped_update(tmp_path):
    trainer = _make_trainer(tmp_path)
    scaler = _SkipScaler()
    trainer.scaler = scaler
    trainer.save = lambda: None
    trainer._evaluate = lambda: None

    trainer.train(steps=1, val_freq=2, save_freq=10)

    assert scaler.attempts == 2
    assert trainer.global_step == 1


def test_training_stops_after_repeated_skipped_updates(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.scaler = _AlwaysSkipScaler()
    trainer.save = lambda: None

    with pytest.raises(FloatingPointError, match="16 consecutive"):
        trainer.train(steps=1, val_freq=2, save_freq=10)
    assert trainer.global_step == 0


def test_training_rejects_non_finite_loss(tmp_path):
    trainer = _make_trainer(tmp_path)

    class _BadLoss(nn.Module):
        def forward(self, out, batch, **kwargs):
            loss = out.logits[-1].sum() * float("nan")
            return {"loss": loss}, loss

    trainer.criterion = _BadLoss()
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        trainer.step()


def test_best_selection_uses_native_size_miou(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.validator.validate = lambda: {"miou": 0.8, "mdice": 0.85}
    trainer.train(steps=1, val_freq=1, save_freq=10)
    state = torch.load(tmp_path / "weights" / "best_miou.pth", weights_only=True)
    assert state["num_classes"] == 4
    assert state["selection"]["metrics"]["miou"] == 0.8


def test_validation_preserves_class_three(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((8, 12), 3, dtype=np.uint8)).save(mask_path)
    trainer = _make_trainer(tmp_path, preprocess=PreprocessSpec())
    trainer.validator.pairs = [(str(image_path), str(mask_path))]
    with torch.no_grad():
        trainer.model.decoder.weight.zero_()
        trainer.model.decoder.bias.zero_()
        trainer.model.decoder.bias[3] = 20
    metrics = trainer.validator.validate()
    assert metrics["miou"] == 1.0
    assert metrics["class_3_recall"] == 1.0


def test_training_validates_final_non_frequency_step(tmp_path):
    trainer = _make_trainer(tmp_path)
    validated = []
    trainer._evaluate = lambda: validated.append(trainer.global_step)
    trainer.save = lambda: None

    trainer.train(steps=1, val_freq=2, save_freq=10)

    assert validated == [1]


@pytest.mark.parametrize(
    ("val_freq", "expected"),
    [(1, [(1, 1.0), (2, 2.0)]),
     (3, [(1, float("-inf")), (2, float("-inf")), (2, 2.0)])],
)
def test_final_save_reuses_current_checkpoint_but_keeps_final_selection(
    tmp_path, val_freq, expected,
):
    trainer = _make_trainer(tmp_path)
    saved = []
    trainer.save = lambda: saved.append((trainer.global_step, trainer.best_region))

    def evaluate():
        trainer.best_region = float(trainer.global_step)

    trainer._evaluate = evaluate
    trainer.train(steps=2, val_freq=val_freq, save_freq=1)

    assert saved == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [("steps", 0), ("val_freq", 0), ("save_freq", 0)],
)
def test_training_rejects_non_positive_frequencies(tmp_path, name, value):
    trainer = _make_trainer(tmp_path)
    kwargs = {"steps": 1, "val_freq": 2, "save_freq": 2}
    kwargs[name] = value
    with pytest.raises(ValueError, match="positive"):
        trainer.train(**kwargs)


@pytest.mark.parametrize("name", ["steps", "val_freq", "save_freq"])
@pytest.mark.parametrize("value", [1.5, True, float("nan"), float("inf")])
def test_training_rejects_non_integer_frequencies_before_logging(
    monkeypatch, tmp_path, name, value,
):
    trainer = _make_trainer(tmp_path)

    def fail(**kwargs):
        raise AssertionError("Invalid settings must fail before opening logs")

    monkeypatch.setattr("src.train.trainer.SummaryWriter", fail)
    kwargs = {"steps": 1, "val_freq": 2, "save_freq": 2, name: value}
    with pytest.raises(ValueError, match=name):
        trainer.train(**kwargs)


@pytest.mark.parametrize("accum_steps", [0, -1, True, 1.5])
def test_trainer_rejects_invalid_accumulation(tmp_path, accum_steps):
    with pytest.raises(ValueError, match="accum_steps"):
        _make_trainer(tmp_path, accum_steps=accum_steps)


@pytest.mark.parametrize("max_grad_norm", [0.0, -1.0, float("nan"), float("inf")])
def test_trainer_rejects_invalid_gradient_limit(tmp_path, max_grad_norm):
    with pytest.raises(ValueError, match="max_grad_norm"):
        _make_trainer(tmp_path, max_grad_norm=max_grad_norm)


def test_native_prediction_uses_saved_preprocess(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 12), dtype=np.uint8)).save(mask_path)

    trainer = _make_trainer(
        tmp_path,
        preprocess=PreprocessSpec(is_sem=True),
    )
    trainer.validator.pairs = [(str(image_path), str(mask_path))]
    captured = {}

    def fake_predict(model, image, **kwargs):
        captured.update(kwargs)
        return np.zeros((4, *image.shape[:2]), dtype=np.float32)

    trainer.validator.predictor = fake_predict
    trainer.validator.validate()

    assert captured == {"size": 1024, "is_sem": True}


def test_saves_ema_model_without_changing_live_weights(tmp_path):
    trainer = _make_trainer(tmp_path)
    initial = trainer.model.decoder.weight.detach().clone()
    _, updated = trainer.step()
    assert updated
    current = trainer.model.decoder.weight.detach().clone()
    expected = initial * 0.99 + current * 0.01
    trainer.save()
    saved = torch.load(tmp_path / "weights/last_ema.pth", weights_only=True)
    torch.testing.assert_close(saved["model"]["decoder.weight"], expected)
    torch.testing.assert_close(trainer.model.decoder.weight, current)
    assert saved["preprocess"]["size"] == 1024
    assert "optimizer" not in saved


def test_resume_requires_ema_state(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.step()
    trainer.save()
    path = tmp_path / "weights/last.train.pth"
    saved = torch.load(path, weights_only=True)
    del saved["ema"]
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="Unsupported training checkpoint"):
        _make_trainer(tmp_path).load_resume(str(path))


@pytest.mark.parametrize("decay", [float("nan"), float("inf"), -0.1, 1.0])
def test_resume_rejects_invalid_ema_decay_before_loading_weights(tmp_path, decay):
    trainer = _make_trainer(tmp_path)
    trainer.step()
    trainer.save()
    path = tmp_path / "weights/last.train.pth"
    saved = torch.load(path, weights_only=True)
    saved["ema_decay"] = decay
    torch.save(saved, path)
    resumed = _make_trainer(tmp_path)
    before = {name: value.clone() for name, value in resumed.model.state_dict().items()}

    with pytest.raises(RuntimeError, match="EMA decay"):
        resumed.load_resume(str(path))

    assert resumed.global_step == 0
    assert not resumed.optimizer.state
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, before[name])



def test_ema_does_not_add_training_forward_passes(tmp_path):
    trainer = _make_trainer(tmp_path, accum_steps=2)
    trainer.model.calls = 0
    _, updated = trainer.step()
    assert updated and trainer.model.calls == 2


def test_teacher_share_waits_then_ramps(tmp_path):
    trainer = _make_trainer(tmp_path, teacher_weight=0.3, teacher_start=10)
    shares = []
    for step in (0, 10, 15, 20, 100):
        trainer.global_step = step
        shares.append(trainer.teacher_share())
    assert shares == pytest.approx([0.0, 0.0, 0.15, 0.3, 0.3])


def test_teacher_uses_ema_weights_and_feeds_the_loss(tmp_path):
    trainer = _make_trainer(tmp_path, teacher_weight=0.3, teacher_start=1)
    seen = []
    trainer.criterion.forward = lambda out, batch, **kwargs: (seen.append(kwargs), _DummyCriterion.forward(trainer.criterion, out, batch))[1]
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        trainer.ema.params["decoder.weight"].zero_()
        trainer.ema.params["decoder.bias"].copy_(torch.tensor([0.0, 9.0, 0.0, 0.0]))
    probs = trainer.teach(x, trainer.model.prepare(x))
    assert torch.all(probs.argmax(1) == 1)
    assert trainer.model.training
    assert trainer.model.decoder.bias.abs().max() < 9
    trainer.global_step = 2
    trainer.step()
    assert seen[-1]["teacher_weight"] == pytest.approx(0.3)
    assert seen[-1]["teacher"].shape == (2, 4, 8, 8)


@pytest.mark.parametrize("options", [{"teacher_weight": 1.0}, {"teacher_weight": -0.1},
                                     {"teacher_start": 0}, {"teacher_start": 1.5}])
def test_teacher_settings_are_checked(tmp_path, options):
    with pytest.raises(ValueError, match="teacher"):
        _make_trainer(tmp_path, **options)
