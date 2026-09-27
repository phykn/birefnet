from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.build.data import build as build_data
from src.build.model import build_predictor
from src.build.trainer import build as build_trainer
from src.config import ROOT
from src.run import load_run
from src.prepare.spec import PreprocessSpec
from test_model import tiny_model


def test_sample_palette_pairs_and_cpu_training_resume(tmp_path, tiny_model, monkeypatch):
    cfg, _, _ = load_run()
    cfg.data.image_dir = str(ROOT / "data/image")
    cfg.data.mask_dir = str(ROOT / "data/mask")
    monkeypatch.setattr(PreprocessSpec, "size", 64)
    cfg.data.min_crop_size = 16
    cfg.loader.num_workers = 0
    cfg.loader.persistent_workers = False
    cfg.loader.pin_memory = False
    cfg.train.steps = 2
    cfg.train.warmup_steps = 0
    cfg.train.ema_decay = 0.9
    for path in sorted(Path(cfg.data.mask_dir).glob("*.png")):
        with Image.open(path) as mask:
            assert mask.mode == "P"
            assert set(np.unique(np.asarray(mask))) == {0, 1, 2, 3}
    train, valid, splits = build_data(cfg)
    assert len(train.dataset) == 3 and len(valid.dataset) == 1
    assert next(iter(train))["mask"].dtype == torch.long
    trainer = build_trainer(cfg, tiny_model, train, valid, save_dir=tmp_path)
    assert trainer.ema.decay == 0.9
    assert [group["name"] for group in trainer.optimizer.param_groups] == ["decoder"]
    before = tiny_model.decoder.conv_out1[0].weight.detach().clone()
    frozen = tiny_model.bb.patch_embed.weight.detach().clone()
    trainer.train(steps=1, val_freq=1, save_freq=1)
    assert not torch.equal(before, tiny_model.decoder.conv_out1[0].weight)
    assert torch.equal(frozen, tiny_model.bb.patch_embed.weight)
    assert (tmp_path / "weights/best_miou.pth").exists()
    resumed = build_trainer(cfg, tiny_model, train, valid, save_dir=tmp_path)
    resumed.load_resume(str(tmp_path / "weights/last.train.pth"))
    assert resumed.global_step == 1
    assert resumed.ema.decay == 0.9
    assert resumed.ema.state_dict().keys() == trainer.ema.state_dict().keys()

    averaged = torch.load(tmp_path / "weights/last_ema.pth", weights_only=True)
    ema_state = resumed.ema.state_dict()
    for name, value in ema_state.items():
        torch.testing.assert_close(averaged["model"][name], value)

    cfg.birefnet.channels = [32, 16, 8, 4]
    image = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        expected = tiny_model.eval()(image).logits[-1]
        for name in ("last.pth", "last.train.pth", "best_miou.pth"):
            model = build_predictor(cfg, str(tmp_path / "weights" / name), torch.device("cpu"))
            torch.testing.assert_close(model(image).logits[-1], expected, rtol=0, atol=0)
            assert PreprocessSpec.from_meta(model.loaded_meta) == trainer.preprocess
            assert not {"optimizer", "scheduler", "scaler", "ema"} & model.loaded_meta.keys()


def test_high_resolution_half_logits_have_finite_dice_gradient():
    from src.model.output import Output
    from src.train.objective import TrainLoss
    logits = torch.ones(1, 4, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 256, 256, dtype=torch.long)
    _, loss = TrainLoss()(Output(logits=[logits]), {"mask": target})
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_ema_checkpoint_loads_for_inference(tmp_path, tiny_model):
    from src.train.ema import EMA
    from src.model.checkpoint import pack_model
    from src.train.checkpoint import atomic_torch_save

    ema = EMA(tiny_model)
    payload = pack_model(tiny_model, PreprocessSpec())
    payload["model"].update(ema.state_dict())
    path = tmp_path / "last_ema.pth"
    atomic_torch_save(payload, path)
    cfg, _, _ = load_run()
    cfg.birefnet.channels = [32, 16, 8, 4]
    restored = build_predictor(cfg, str(path), torch.device("cpu"))
    with torch.inference_mode():
        assert restored(torch.randn(1, 3, 64, 64)).logits[-1].shape == (1, 4, 64, 64)
