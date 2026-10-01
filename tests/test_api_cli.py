from argparse import Namespace
import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image

import run_api
from src.model import BiRefNet
from src.model.checkpoint import pack_model
from src.predict.model import load_run_config
from src.prepare.spec import PreprocessSpec
from test_model import tiny_model


@pytest.mark.parametrize("explicit", [False, True])
def test_api_selects_saved_config_unless_explicitly_overridden(monkeypatch, tmp_path, explicit):
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = weights / "last.pth"
    checkpoint.touch()
    OmegaConf.save({"birefnet": {"num_classes": 5}}, tmp_path / "config.yaml")
    config = tmp_path / "custom.yaml"
    OmegaConf.save({"birefnet": {"num_classes": 6}}, config)
    args = Namespace(weight=str(checkpoint), config=str(config) if explicit else None,
                     device="cpu", host="127.0.0.1", port=8000)
    captured = {}

    def load_model(cfg, path, device):
        captured.update(num_classes=cfg.birefnet.num_classes, path=path, device=device.type)
        return SimpleNamespace(num_classes=cfg.birefnet.num_classes)

    monkeypatch.setattr(run_api, "parse_args", lambda: args)
    monkeypatch.setattr(run_api, "load_model", load_model)
    monkeypatch.setattr(run_api, "build_app", lambda **kwargs: kwargs)
    monkeypatch.setattr(run_api.uvicorn, "run", lambda app, **kwargs: captured.update(server=kwargs))

    run_api.main()

    assert captured["num_classes"] == (6 if explicit else 5)
    assert captured["path"] == str(checkpoint)
    assert captured["device"] == "cpu"
    assert captured["server"] == {"host": "127.0.0.1", "port": 8000}


def test_prediction_config_falls_back_to_model_defaults(tmp_path):
    cfg = load_run_config(tmp_path / "weights/last.pth")
    assert cfg.birefnet.num_classes == 4


def test_api_serves_nondefault_checkpoint_using_saved_config(monkeypatch, tmp_path, tiny_model):
    monkeypatch.setattr(PreprocessSpec, "size", 64)
    model = BiRefNet(channels=[32, 16, 8, 4], num_classes=5)
    with torch.no_grad():
        model.decoder.conv_out1[0].weight.zero_()
        model.decoder.conv_out1[0].bias.zero_()
        model.decoder.conv_out1[0].bias[4] = 10
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = weights / "last.pth"
    torch.save(pack_model(model, PreprocessSpec(is_sem=True), 254), checkpoint)
    OmegaConf.save({"birefnet": {"channels": [32, 16, 8, 4], "num_classes": 5,
                                 "weight": "absent_base.pth"}}, tmp_path / "config.yaml")
    monkeypatch.setattr(run_api, "parse_args", lambda: Namespace(
        weight=str(checkpoint), config=None, device="cpu", host="127.0.0.1", port=8000))
    captured = {}
    monkeypatch.setattr(run_api.uvicorn, "run", lambda app, **kwargs: captured.update(app=app))

    run_api.main()

    source = BytesIO()
    Image.new("RGB", (29, 17), (100, 120, 140)).save(source, format="PNG")
    with TestClient(captured["app"]) as client:
        assert client.app.state.preprocess == PreprocessSpec(is_sem=True)
        response = client.post("/predict", json={
            "base64_str": base64.b64encode(source.getvalue()).decode("ascii"),
            "tiles": [1, 2],
        })
    assert response.status_code == 200
    payload = response.json()
    assert payload["num_classes"] == 5
    with Image.open(BytesIO(base64.b64decode(payload["base64_str"]))) as labels:
        assert labels.mode == "P"
        assert labels.size == (29, 17)
        np.testing.assert_array_equal(np.asarray(labels), np.full((17, 29), 4))
