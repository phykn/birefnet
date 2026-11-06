import torch.nn as nn
import torch.nn.functional as F


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, 1)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(4, 4, 1),
                nn.Conv2d(4, 8, 1),
                nn.Conv2d(8, 16, 1),
                nn.Conv2d(16, 32, 1),
            ]
        )
        for idx, channels in enumerate([4, 8, 16, 32]):
            setattr(self, f"norm{idx}", nn.BatchNorm2d(channels))

    def forward(self, x):
        x = self.patch_embed(F.avg_pool2d(x, 4))
        outs = []
        for idx, layer in enumerate(self.layers):
            if idx:
                x = F.avg_pool2d(x, 2)
            x = getattr(self, f"norm{idx}")(layer(x))
            outs.append(x)
        return outs
