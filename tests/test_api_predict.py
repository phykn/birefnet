import base64
from io import BytesIO
from PIL import Image
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from src.prepare.spec import PreprocessSpec
from backend.codec import ImageLimitError, decode

PREDICT_TARGET = "backend.route.predict_mask"


def _encode(image: np.ndarray) -> str:
    source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    return base64.b64encode(encoded).decode("ascii")


def _decode(value: str) -> np.ndarray:
    with Image.open(BytesIO(base64.b64decode(value))) as image:
        return np.asarray(image).copy()


def test_defaults_to_class_labels(monkeypatch, api_client):
    captured = {}
    def fake_predict(model, image, **kwargs):
        captured.update(kwargs)
        labels = np.tile(np.arange(4, dtype=np.uint8), (16, 6))
        return labels
    monkeypatch.setattr(PREDICT_TARGET, fake_predict)
    client = api_client(SimpleNamespace(num_classes=4), torch.device("cpu"))
    response = client.post("/predict", json={"id": "sample", "base64_str": _encode(np.zeros((16, 24, 3), np.uint8)), "tiles": [1, 3]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "sample"
    assert payload["output_mode"] == "labels"
    assert payload["num_classes"] == 4
    assert payload["value_range"] == [0, 3]
    assert payload["class_id"] is None
    assert payload["dtype"] == "uint8"
    with Image.open(BytesIO(base64.b64decode(payload["base64_str"]))) as png:
        assert png.mode == "P"
        assert set(np.unique(np.asarray(png))) == {0, 1, 2, 3}
    np.testing.assert_array_equal(_decode(payload["base64_str"]), np.tile(np.arange(4, dtype=np.uint8), (16, 6)))
    assert captured["tiles"] == (1, 3)
    assert captured["size"] == 1024
    assert captured["is_sem"] is False


def test_probability_mode_requires_class_id(monkeypatch, api_client):
    def fake_predict(model, image, **kwargs):
        assert kwargs["output_mode"] == "probability"
        assert kwargs["class_id"] == 3
        return np.full(image.shape[:2], 127, dtype=np.uint8)
    monkeypatch.setattr(PREDICT_TARGET, fake_predict)
    client = api_client(SimpleNamespace(num_classes=4), torch.device("cpu"))
    response = client.post("/predict", json={"base64_str": _encode(np.zeros((8, 8, 3), np.uint8)), "output_mode": "probability", "class_id": 3, "tiles": [1, 3]})
    assert response.status_code == 200
    assert response.json()["class_id"] == 3
    assert response.json()["value_range"] == [0, 255]
    assert np.all(_decode(response.json()["base64_str"]) == 127)


def test_rejects_invalid_image(monkeypatch, api_client):
    monkeypatch.setattr(
        PREDICT_TARGET,
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    client = api_client(SimpleNamespace(num_classes=4), torch.device("cpu"))
    response = client.post("/predict", json={"base64_str": "invalid"})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid image"}


@pytest.mark.parametrize("fields", [
    {"threshold": 0.5}, {"other": True}, {"output_mode": "binary"},
    {"output_mode": "probability"}, {"output_mode": "probability", "class_id": 4},
    {"output_mode": "probability", "class_id": -1},
    {"output_mode": "probability", "class_id": True}, {"class_id": 1},
    {"tiles": [0]}, {"overlap": 0.2},
])
def test_rejects_invalid_output_contract(api_client, fields):
    client = api_client(SimpleNamespace(num_classes=4), torch.device("cpu"))
    response = client.post("/predict", json={"base64_str": _encode(np.zeros((4, 4, 3), np.uint8)), **fields})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("name", "limit"),
    [("MAX_BYTES", 1), ("MAX_PIXELS", 15)],
)
def test_rejects_large_image(monkeypatch, api_client, name, limit):
    monkeypatch.setattr(
        f"backend.codec.{name}",
        limit,
    )
    monkeypatch.setattr(
        PREDICT_TARGET,
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    client = api_client(SimpleNamespace(num_classes=4), torch.device("cpu"))
    response = client.post(
        "/predict",
        json={"base64_str": _encode(np.zeros((4, 4, 3), np.uint8))},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "image too large"}


def test_uses_checkpoint_preprocess_contract(monkeypatch, api_client):
    captured = {}

    def fake_predict(model, image, **kwargs):
        captured.update(kwargs)
        return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(PREDICT_TARGET, fake_predict)
    client = api_client(
        SimpleNamespace(num_classes=4),
        torch.device("cpu"),
        preprocess=PreprocessSpec(is_sem=True),
    )
    response = client.post(
        "/predict",
        json={"base64_str": _encode(np.zeros((4, 4, 3), np.uint8))},
    )

    assert response.status_code == 200
    assert captured["size"] == 1024
    assert captured["is_sem"] is True


def test_decode_rejects_oversized_base64_before_allocating(monkeypatch):
    monkeypatch.setattr("backend.codec.MAX_BASE64_LENGTH", 4)
    with pytest.raises(ImageLimitError):
        decode("A" * 8)


def test_reads_preprocess_from_full_checkpoint_metadata():
    from backend.app import read_preprocess
    model = SimpleNamespace(loaded_meta={"preprocess": {"size": 1024, "is_sem": True}})
    assert read_preprocess(model) == PreprocessSpec(is_sem=True)


def test_app_defaults_to_loaded_checkpoint_preprocess(api_client):
    model = SimpleNamespace(num_classes=4, loaded_meta={
        "preprocess": {"size": 1024, "is_sem": True},
    })
    client = api_client(model, torch.device("cpu"))
    assert client.app.state.preprocess == PreprocessSpec(is_sem=True)


def test_app_explicit_preprocess_overrides_checkpoint(api_client):
    model = SimpleNamespace(num_classes=4, loaded_meta={
        "preprocess": {"size": 1024, "is_sem": True},
    })
    client = api_client(model, torch.device("cpu"), preprocess=PreprocessSpec())
    assert client.app.state.preprocess == PreprocessSpec()


def test_rejects_pillow_decompression_bomb_as_large_image(monkeypatch, api_client):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 7)
    client = api_client(SimpleNamespace(num_classes=4), torch.device("cpu"))
    response = client.post(
        "/predict", json={"base64_str": _encode(np.zeros((4, 4, 3), np.uint8))},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "image too large"}
