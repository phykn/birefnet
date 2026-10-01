import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.check_data import inspect
from scripts.predict import run
from src.model.output import Output
from test_model import tiny_model


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.logits = torch.nn.Parameter(torch.tensor([0., 0., 0., 10.]))
        self.loaded_meta = {"preprocess": {"size": 1024, "is_sem": True}}

    def forward(self, image):
        return Output([self.logits[None, :, None, None].expand(len(image), 4, *image.shape[-2:])])


def test_loader_preview_excludes_padding_and_ignored_labels(tmp_path):
    masks = torch.tensor([[[0, 1], [3, 255]]])
    batch = {"image": torch.zeros(1, 3, 2, 2),
             "mask": masks, "valid": torch.tensor([[[[1, 1], [0, 1]]]]).float(),
             "cut": torch.zeros(1, 1, 2, 2)}
    report = inspect([batch, batch], 4, 255, 1, tmp_path)
    assert report["class_pixels"] == [1, 1, 0, 0]
    assert len(report["batches"]) == 1
    assert report["batches"][0]["ignored_pixels"] == 2
    assert report == json.loads((tmp_path / "summary.json").read_text())
    with Image.open(report["batches"][0]["previews"][0]) as preview:
        assert preview.size == (1600, 352)


def test_loader_preview_rejects_empty_or_invalid_batches(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        inspect([], 4, 255, 1, tmp_path)
    with pytest.raises(ValueError, match="positive"):
        inspect([], 4, 255, 0, tmp_path)


def test_prediction_saves_native_class_ids_and_probability(tmp_path):
    image = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((17, 29, 3), 120, dtype=np.uint8)).save(image)
    Image.fromarray(np.full((17, 29), 3, dtype=np.uint8)).save(mask)
    output = tmp_path / "out"
    report = run(ConstantModel(), image, output, tiles=(1, 2), mask_path=mask, class_id=3)
    with Image.open(output / "labels.png") as labels:
        assert labels.mode == "P"
        assert labels.size == (29, 17)
        assert np.all(np.asarray(labels) == 3)
    with Image.open(output / "probability_3.png") as prob:
        assert np.all(np.asarray(prob) == 255)
    assert report["class_pixels"] == [0, 0, 0, 17 * 29]
    assert report["preprocess"] == {"size": 1024, "is_sem": True}
    assert (output / "preview.png").is_file()
    assert (output / "overlay.png").is_file()


def test_prediction_rejects_bad_class_before_reading_image(tmp_path):
    with pytest.raises(ValueError, match="class_id"):
        run(ConstantModel(), Path("missing.png"), tmp_path, class_id=4)


@pytest.mark.parametrize("stored, override", [(254, None), (254, 253), (None, None)])
def test_prediction_uses_checkpoint_ignore_index_for_ground_truth(tmp_path, stored, override):
    model = ConstantModel()
    if stored is not None:
        model.loaded_meta["ignore_index"] = stored
    image, mask = tmp_path / "input.png", tmp_path / "mask.png"
    Image.new("RGB", (12, 8)).save(image)
    ignored = override if override is not None else (stored if stored is not None else 255)
    Image.new("L", (12, 8), ignored).save(mask)

    report = run(model, image, tmp_path / "out", mask_path=mask, ignore_index=override)

    assert report["shape"] == [8, 12]
    assert (tmp_path / "out/preview.png").is_file()


def test_prediction_rejects_misaligned_ground_truth(tmp_path):
    image, mask = tmp_path / "input.png", tmp_path / "mask.png"
    Image.new("RGB", (12, 8)).save(image)
    Image.new("L", (8, 12)).save(mask)
    with pytest.raises(ValueError, match="dimensions"):
        run(ConstantModel(), image, tmp_path / "out", mask_path=mask)


def test_prediction_cli_loads_saved_run_configuration(monkeypatch, tmp_path, tiny_model):
    from omegaconf import OmegaConf
    from scripts.predict import main
    from src.model.checkpoint import pack_model
    from src.prepare.spec import PreprocessSpec

    monkeypatch.setattr(PreprocessSpec, "size", 64)
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = weights / "last.pth"
    torch.save(pack_model(tiny_model, PreprocessSpec()), checkpoint)
    OmegaConf.save({"birefnet": {"channels": [32, 16, 8, 4], "num_classes": 4}}, tmp_path / "config.yaml")
    image = tmp_path / "image.png"
    Image.new("RGB", (29, 17), (100, 120, 140)).save(image)
    output = tmp_path / "result"
    monkeypatch.setattr("sys.argv", ["predict.py", "--weight", str(checkpoint),
                                    "--image", str(image), "--output", str(output)])
    main()
    with Image.open(output / "labels.png") as labels:
        assert labels.size == (29, 17)
        assert np.asarray(labels).max() < 4
