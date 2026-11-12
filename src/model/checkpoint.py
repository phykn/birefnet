from dataclasses import asdict
from pathlib import Path

import torch

from ..config import parse_config, resolve
from .net import BiRefNet

TRAINED = ("squeeze_module.", "decoder.")
SEGMENTATION_HEADS = (
    "decoder.conv_out1.0",
    "decoder.conv_ms_spvn_2",
    "decoder.conv_ms_spvn_3",
    "decoder.conv_ms_spvn_4",
)


def build_model(cfg):
    model = BiRefNet(num_classes=cfg.model.classes, auxiliary=cfg.loss.aux > 0)
    base = torch.load(resolve(cfg.model.weight), map_location="cpu", weights_only=True)
    load_base(model, base)
    return model


def load_base(model, base):
    expected = model.state_dict()
    if base.keys() != expected.keys():
        diff = sorted(base.keys() ^ expected.keys())[:10]
        raise RuntimeError(f"Base checkpoint keys do not match BiRefNet: {diff}")
    state = {}
    for key, value in base.items():
        shape = expected[key].shape
        is_new_head = key.rsplit(".", 1)[0] in SEGMENTATION_HEADS and value.shape == (
            1,
            *shape[1:],
        )
        if value.shape != shape and not is_new_head:
            raise RuntimeError(
                f"Incompatible base checkpoint tensor: {key}: "
                f"{tuple(value.shape)} != {tuple(shape)}"
            )
        state[key] = expected[key] if is_new_head else value
    model.load_state_dict(state)


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(parse_config(state["config"]))
    load_trained(model, state["model"])
    return model.to(device).eval()


def load_trained(model, state):
    expected = set(select_trained(model.state_dict()))
    if set(state) != expected:
        diff = sorted(set(state) ^ expected)[:10]
        raise RuntimeError(f"Checkpoint trained keys differ: {diff}")
    model.load_state_dict(state, strict=False)


def select_trained(state):
    return {key: value for key, value in state.items() if key.startswith(TRAINED)}


def pack_checkpoint(model, cfg, step, metrics):
    trained = select_trained(model.state_dict())
    return {
        "config": asdict(cfg),
        "step": step,
        "metrics": metrics,
        "model": {key: value.detach().cpu().clone() for key, value in trained.items()},
    }


def write_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
