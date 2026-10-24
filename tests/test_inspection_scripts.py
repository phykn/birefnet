from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.model.output import Output
from test_model import tiny_model

run = import_module("scripts.03_predict").run
figure = import_module("scripts._draw").figure


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.logits = torch.nn.Parameter(torch.tensor([0., 0., 0., 10.]))
        self.loaded_meta = {"preprocess": {"size": 1024, "is_sem": True}}

    def forward(self, image):
        return Output([self.logits[None, :, None, None].expand(len(image), 4, *image.shape[-2:])])


def test_dataset_cli_shows_random_pairs_from_config(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from sample_data import write

    write(tmp_path / "data")
    config = tmp_path / "train.yaml"
    OmegaConf.save({"data": {"image_dir": str(tmp_path / "data/image"),
                             "mask_dir": str(tmp_path / "data/mask")}}, config)
    import matplotlib.pyplot as plt
    from src.config import load_config

    script = import_module("scripts.01_dataset")
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(
        sum(1 for axis in plt.gcf().axes if axis.images)))
    monkeypatch.setattr(script, "load_config", lambda: load_config(config))
    script.main()
    assert shown == [2 * 3]


def test_prediction_returns_native_class_map_and_probability(tmp_path):
    image = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((17, 29, 3), 120, dtype=np.uint8)).save(image)
    Image.fromarray(np.full((17, 29), 3, dtype=np.uint8)).save(mask)
    labels, panels = run(ConstantModel(), image, tiles=(1, 2), mask_path=mask, class_id=3)
    assert labels.shape == (17, 29) and np.all(labels == 3)
    assert len(panels) == 5
    assert panels[2].shape == (17, 29, 3)
    assert np.all(panels[4] == 255)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.png", "mask.png"]


def test_prediction_rejects_bad_class_before_reading_image():
    with pytest.raises(ValueError, match="class_id"):
        run(ConstantModel(), Path("missing.png"), class_id=4)


@pytest.mark.parametrize("stored, override", [(254, None), (254, 253), (None, None)])
def test_prediction_uses_checkpoint_ignore_index_for_ground_truth(tmp_path, stored, override):
    model = ConstantModel()
    if stored is not None:
        model.loaded_meta["ignore_index"] = stored
    image, mask = tmp_path / "input.png", tmp_path / "mask.png"
    Image.new("RGB", (12, 8)).save(image)
    ignored = override if override is not None else (stored if stored is not None else 255)
    Image.new("L", (12, 8), ignored).save(mask)

    labels, panels = run(model, image, mask_path=mask, ignore_index=override)

    assert labels.shape == (8, 12)
    assert np.all(panels[1] == 100)


def test_prediction_rejects_misaligned_ground_truth(tmp_path):
    image, mask = tmp_path / "input.png", tmp_path / "mask.png"
    Image.new("RGB", (12, 8)).save(image)
    Image.new("L", (8, 12)).save(mask)
    with pytest.raises(ValueError, match="dimensions"):
        run(ConstantModel(), image, mask_path=mask)


def test_prediction_cli_loads_saved_run_configuration(monkeypatch, tmp_path, tiny_model):
    from omegaconf import OmegaConf
    main = import_module("scripts.03_predict").main
    from src.model.checkpoint import pack_model
    from src.prepare.spec import PreprocessSpec

    monkeypatch.setattr(PreprocessSpec, "size", 64)
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = weights / "last.pth"
    torch.save(pack_model(tiny_model, PreprocessSpec()), checkpoint)
    base = tmp_path / "base.pth"
    torch.save(tiny_model.state_dict(), base)
    OmegaConf.save({"model": {"channels": [32, 16, 8, 4], "classes": 4, "depth": False,
                              "weight": str(base)}}, tmp_path / "config.yaml")
    image = tmp_path / "image.png"
    Image.new("RGB", (29, 17), (100, 120, 140)).save(image)
    shown = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: shown.append(True))
    monkeypatch.setattr("sys.argv", ["03_predict.py", "--weight", str(checkpoint), "--image", str(image)])
    main()
    assert shown == [True]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["base.pth", "config.yaml", "image.png", "weights"]
