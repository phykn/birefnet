from pathlib import Path

import torch
from torch import nn

from dataclasses import asdict
from ..config import parse_config, resolve

FORMAT = "birefnet-multiclass-v3"
# Frozen backbone weights are reconstructed from the local base checkpoint.
TRAINED = ("squeeze_module.", "decoder.")
SEGMENTATION_HEADS = (
    "decoder.conv_out1.0",
    "decoder.conv_ms_spvn_2",
    "decoder.conv_ms_spvn_3",
    "decoder.conv_ms_spvn_4",
)


def load_base(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise RuntimeError("Base checkpoint must be a flat tensor state_dict")
    # Optional depth branches have their own initialization.
    added = ("depth.", "decoder.dpt_blks.")
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(added)
    }
    if state.keys() != expected.keys():
        raise RuntimeError(
            f"Base checkpoint keys do not match BiRefNet: {sorted(state.keys() ^ expected.keys())[:10]}"
        )
    replaced = {}
    for key, value in state.items():
        shape = expected[key].shape
        if value.shape == shape:
            continue
        is_head = key.rsplit(".", 1)[0] in SEGMENTATION_HEADS
        if (
            not is_head
            or value.shape != (1, *shape[1:])
            or shape[0] != model.num_classes
        ):
            raise RuntimeError(
                f"Incompatible base checkpoint tensor: {key}: {tuple(value.shape)} != {tuple(shape)}"
            )
        replaced[key] = expected[key]
    missing, unexpected = model.load_state_dict({**state, **replaced}, strict=False)
    if unexpected or any(not key.startswith(added) for key in missing):
        raise RuntimeError("Base checkpoint keys do not match BiRefNet")


def trained_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if key.startswith(TRAINED)}


def load_trained(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a checkpoint into a model that already holds its base weights."""
    expected = set(trained_state(model.state_dict()))
    if set(state) != expected:
        raise RuntimeError(
            f"Checkpoint trained keys differ: {sorted(set(state) ^ expected)[:10]}"
        )
    model.load_state_dict(state, strict=False)


def write_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    fmt = state.get("format") if isinstance(state, dict) else None
    if fmt != FORMAT:
        if fmt == "birefnet-multiclass-v2":
            raise ValueError(
                "이전 형식 birefnet-multiclass-v2입니다. 커밋 750d7d9의 코드로 읽으세요."
            )
        raise ValueError(f"Unsupported checkpoint format: {fmt}")
    parse_config(state["config"], ignore_unknown=True)
    return state


def build(cfg):
    from .net import BiRefNet

    from .depth import Depth

    depth = Depth.load(resolve(cfg.model.depth_weight)) if cfg.model.depth else None
    model = BiRefNet(
        num_classes=cfg.model.classes, depth=depth, auxiliary=cfg.loss.aux > 0
    )
    load_base(
        model,
        torch.load(resolve(cfg.model.weight), map_location="cpu", weights_only=True),
    )
    return model


def load(path, device):
    state = read_checkpoint(path)
    cfg = parse_config(state["config"], ignore_unknown=True)
    model = build(cfg)
    load_trained(model, state["model"])
    return model.to(device).eval()


def pack_model(model, cfg, step, metrics):
    return dict(
        format=FORMAT,
        config=asdict(cfg),
        step=step,
        metrics=metrics,
        model={
            k: v.detach().cpu().clone()
            for k, v in trained_state(model.state_dict()).items()
        },
    )
