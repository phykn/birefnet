from collections.abc import Mapping

import torch
import torch.nn as nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self._source_model = model
        self._current_params = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        saved_keys = model.state_dict().keys()
        self._current_params.update(
            (name, value) for name, value in model.named_buffers() if name in saved_keys
        )
        self.params = {
            name: param.detach().float().clone() if param.is_floating_point() else param.detach().clone()
            for name, param in self._current_params.items()
        }
        if not self.params:
            raise RuntimeError("EMA requires model state")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = (
            self._current_params
            if model is self._source_model
            else {**dict(model.named_parameters()), **dict(model.named_buffers())}
        )
        for name, avg in self.params.items():
            param = current[name].detach().to(device=avg.device, dtype=avg.dtype)
            if avg.is_floating_point():
                avg.lerp_(param, 1.0 - self.decay)
            else:
                avg.copy_(param)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.params.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.params)
        loaded = set(state)
        if expected != loaded:
            missing = sorted(expected - loaded)
            unexpected = sorted(loaded - expected)
            raise RuntimeError(
                "EMA state keys do not match the model: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for name, avg in self.params.items():
            value = state[name]
            if tuple(value.shape) != tuple(avg.shape):
                raise RuntimeError(f"EMA state shape does not match for {name}")
            avg.copy_(value.to(device=avg.device, dtype=avg.dtype))
