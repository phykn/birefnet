import torch

from src.config import load_config
from src.model import Output
from src.train.loss import TrainLoss
from src.train.schedule import cosine_schedule
from src.train.trainer import Trainer


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Conv2d(3, 4, 1)
        self.num_classes = 4

    def forward(self, x):
        return Output([self.decoder(x)])

    def list_trainable(self):
        return list(self.parameters())

    @property
    def stats(self):
        count = sum(param.numel() for param in self.parameters())
        return {"total": count, "trainable": count, "frozen": 0}


def trainer(tmp_path, model=None):
    cfg = load_config()
    cfg.train.steps = 4
    cfg.train.warmup = 0
    cfg.train.accum = 1
    m = model or Tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.train.lr)
    sch = cosine_schedule(opt, 4, cfg.train.min_lr)
    batches = [
        {
            "image": torch.rand(1, 3, 16, 16),
            "mask": torch.zeros(1, 16, 16, dtype=torch.long),
        }
    ]
    return Trainer(m, batches, [], TrainLoss(), opt, sch, tmp_path, cfg)

