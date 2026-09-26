from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.build.data import build as build_data
from src.build.model import build_predictor
from src.build.trainer import build as build_trainer
from src.config import ROOT, load_run
from src.prepare.spec import PreprocessSpec
from test_model import tiny_model


def test_sample_palette_pairs_and_cpu_training_resume(tmp_path, tiny_model):
    cfg, _, _ = load_run()
    cfg.data.image_dir = str(ROOT / "data/image")
    cfg.data.mask_dir = str(ROOT / "data/mask")
    cfg.data.size = 64
    cfg.loader.num_workers = 0
    cfg.loader.persistent_workers = False
    cfg.loader.pin_memory = False
    cfg.train.steps = 2
    cfg.train.warmup_steps = 0
    cfg.teacher.enabled = False
    for path in sorted(Path(cfg.data.mask_dir).glob("*.png")):
        with Image.open(path) as mask:
            assert mask.mode == "P"
            assert set(np.unique(np.asarray(mask))) == {0, 1, 2, 3}
    train, valid, splits = build_data(cfg)
    assert len(train.dataset) == 3 and len(valid.dataset) == 1
    assert next(iter(train))["mask"].dtype == torch.long
    trainer = build_trainer(cfg, tiny_model, train, valid, save_dir=tmp_path)
    before = tiny_model.decoder.conv_out1[0].weight.detach().clone()
    frozen = tiny_model.bb.patch_embed.weight.detach().clone()
    trainer.train(steps=1, val_freq=1, save_freq=1)
    assert not torch.equal(before, tiny_model.decoder.conv_out1[0].weight)
    assert torch.equal(frozen, tiny_model.bb.patch_embed.weight)
    assert (tmp_path / "weights/best_miou.pth").exists()
    resumed = build_trainer(cfg, tiny_model, train, valid, save_dir=tmp_path)
    resumed.load_resume(str(tmp_path / "weights/last.train.pth"))
    assert resumed.global_step == 1
    assert resumed.teacher is None

    cfg.birefnet.channels = [32, 16, 8, 4]
    image = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        expected = tiny_model.eval()(image).logits[-1]
        for name in ("last.pth", "last.train.pth", "best_miou.pth"):
            model = build_predictor(cfg, str(tmp_path / "weights" / name), torch.device("cpu"))
            torch.testing.assert_close(model(image).logits[-1], expected, rtol=0, atol=0)
            assert PreprocessSpec.from_meta(model.loaded_meta) == trainer.preprocess


def test_high_resolution_half_logits_have_finite_dice_gradient():
    from src.model.output import Output
    from src.train.objective import TrainLoss
    logits = torch.ones(1, 4, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 256, 256, dtype=torch.long)
    _, loss = TrainLoss()(Output(logits=[logits]), {"mask": target})
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
