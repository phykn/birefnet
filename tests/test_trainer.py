import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.model.output import Output
from src.prepare.spec import PreprocessSpec
from src.predict.inference import predict_logits
from src.train.trainer import Trainer
from src.train.schedule import CosineSchedule


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.conv = nn.Conv2d(3, 4, 1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        pred = self.conv(x)
        if self.training:
            return Output(logits=[pred, pred])
        return Output(logits=[pred])

    def list_trainable(self):
        return list(self.parameters())



class _DummyCriterion(nn.Module):
    def forward(self, out, batch):
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


def _make_trainer(tmp_path, accum_steps=1, preprocess=None):
    model = _DummyModel()
    train_loader = DataLoader(_DummyDataset(4), batch_size=2)
    valid_loader = DataLoader(_DummyDataset(4), batch_size=2)
    criterion = _DummyCriterion()
    optimizer = torch.optim.AdamW(model.list_trainable(), lr=1e-2)
    scheduler = CosineSchedule(
        optimizer=optimizer,
        first_cycle_steps=10,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_steps=2,
    )
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=str(tmp_path),
        predictor=predict_logits,
        max_grad_norm=1.0,
        accum_steps=accum_steps,
        preprocess=preprocess,
    )


def test_trainer_step_returns_loss_dict_and_updates_params(tmp_path):
    trainer = _make_trainer(tmp_path)
    before = trainer.model.conv.weight.detach().clone()

    losses, updated = trainer.step()

    assert updated
    assert {"loss", "seg", "cons", "aux"} <= set(losses.keys())
    assert all(isinstance(v, float) for v in losses.values())
    after = trainer.model.conv.weight
    assert not torch.allclose(before, after)


def test_trainer_step_with_accum(tmp_path):
    trainer = _make_trainer(tmp_path, accum_steps=2)
    losses, updated = trainer.step()
    assert updated
    assert "loss" in losses


def test_trainer_validate_returns_avg_losses(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.model.calls = 0
    metrics = trainer.validate()
    assert "loss" in metrics
    assert "seg" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert trainer.model.calls == len(trainer.valid_loader)


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
    expected_weight = trainer.model.conv.weight.detach().clone()
    expected_ema = trainer.ema.state_dict()
    expected_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.best_region = 0.6
    trainer.save()

    resumed = _make_trainer(tmp_path, preprocess=preprocess)
    resumed.load_resume(os.path.join(tmp_path, "weights", "last.train.pth"))
    assert resumed.global_step == trainer.global_step
    assert resumed.best_region == trainer.best_region
    assert torch.allclose(resumed.model.conv.weight, expected_weight)
    assert resumed.optimizer.param_groups[0]["lr"] == expected_lr
    assert resumed.scheduler.step_in_cycle == trainer.scheduler.step_in_cycle
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
    weight = trainer.model.conv.weight.detach().clone()
    ema = trainer.ema.state_dict()
    cycle = trainer.scheduler.step_in_cycle

    _, updated = trainer.step()

    assert not updated
    assert trainer.global_step == 0
    assert trainer.scheduler.step_in_cycle == cycle
    assert torch.allclose(trainer.model.conv.weight, weight)
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


def test_best_selection_uses_multiclass_deployment_miou(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.validate = lambda: {"loss": 0.0}
    trainer.validate_deploy = lambda: {"deploy_miou": 0.8, "deploy_mdice": 0.85}
    trainer.train(steps=1, val_freq=1, save_freq=10)
    state = torch.load(tmp_path / "weights" / "best_miou.pth", weights_only=True)
    assert state["num_classes"] == 4
    assert state["selection"]["metrics"]["deploy_miou"] == 0.8


def test_deployment_validation_preserves_class_three(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((8, 12), 3, dtype=np.uint8)).save(mask_path)
    trainer = _make_trainer(tmp_path, preprocess=PreprocessSpec())
    trainer.valid_loader.dataset.data = [(str(image_path), str(mask_path))]
    with torch.no_grad():
        trainer.model.conv.weight.zero_()
        trainer.model.conv.bias.zero_()
        trainer.model.conv.bias[3] = 20
    metrics = trainer.validate_deploy()
    assert metrics["deploy_miou"] == 1.0
    assert metrics["deploy_class_3_recall"] == 1.0


def test_training_validates_final_non_frequency_step(tmp_path):
    trainer = _make_trainer(tmp_path)
    validated = []
    trainer._evaluate = lambda: validated.append(trainer.global_step)
    trainer.save = lambda: None

    trainer.train(steps=1, val_freq=2, save_freq=10)

    assert validated == [1]


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


@pytest.mark.parametrize("accum_steps", [0, -1, True, 1.5])
def test_trainer_rejects_invalid_accumulation(tmp_path, accum_steps):
    with pytest.raises(ValueError, match="accum_steps"):
        _make_trainer(tmp_path, accum_steps=accum_steps)


def test_native_prediction_uses_saved_preprocess(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 12), dtype=np.uint8)).save(mask_path)

    trainer = _make_trainer(
        tmp_path,
        preprocess=PreprocessSpec(is_sem=True),
    )
    trainer.valid_loader.dataset.data = [(str(image_path), str(mask_path))]
    captured = {}

    def fake_predict(model, image, **kwargs):
        captured.update(kwargs)
        return np.zeros((4, *image.shape[:2]), dtype=np.float32)

    trainer.predictor = fake_predict
    list(trainer.predict_native(trainer.valid_loader))

    assert captured == {"size": 1024, "is_sem": True}


def test_saves_ema_model_without_changing_live_weights(tmp_path):
    trainer = _make_trainer(tmp_path)
    initial = trainer.model.conv.weight.detach().clone()
    _, updated = trainer.step()
    assert updated
    current = trainer.model.conv.weight.detach().clone()
    expected = initial * 0.99 + current * 0.01
    trainer.save()
    saved = torch.load(tmp_path / "weights/last_ema.pth", weights_only=True)
    torch.testing.assert_close(saved["model"]["conv.weight"], expected)
    torch.testing.assert_close(trainer.model.conv.weight, current)
    assert saved["preprocess"]["size"] == 1024
    assert "optimizer" not in saved


def test_resume_without_ema_starts_average_from_restored_weights(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.step()
    trainer.save()
    path = tmp_path / "weights/last.train.pth"
    saved = torch.load(path, weights_only=True)
    del saved["ema"]
    del saved["ema_decay"]
    torch.save(saved, path)
    resumed = _make_trainer(tmp_path)
    resumed.load_resume(str(path))
    for name, param in resumed.model.named_parameters():
        torch.testing.assert_close(resumed.ema.params[name], param)



def test_ema_does_not_add_training_forward_passes(tmp_path):
    trainer = _make_trainer(tmp_path, accum_steps=2)
    trainer.model.calls = 0
    _, updated = trainer.step()
    assert updated and trainer.model.calls == 2
