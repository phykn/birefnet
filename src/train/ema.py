from contextlib import contextmanager

import torch


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.live = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.params = {n: p.detach().clone() for n, p in self.live.items()}

    @torch.no_grad()
    def update(self, step):
        decay = min(self.decay, (1 + step) / (10 + step))
        for name, avg in self.params.items():
            avg.lerp_(self.live[name], 1 - decay)

    @contextmanager
    def applied(self):
        with torch.no_grad():
            saved = {n: p.clone() for n, p in self.live.items()}
            for name, param in self.live.items():
                param.copy_(self.params[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in self.live.items():
                    param.copy_(saved[name])

    def state_dict(self):
        return {n: p.detach().cpu().clone() for n, p in self.params.items()}

    def load_state_dict(self, state):
        shapes = {n: p.shape for n, p in self.params.items()}
        if {n: v.shape for n, v in state.items()} != shapes:
            raise RuntimeError("EMA state does not match the trainable weights")
        for name, avg in self.params.items():
            avg.copy_(state[name])
