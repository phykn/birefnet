from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.image import read_image, read_mask
from ..prepare.spec import PreprocessSpec
from .metrics import confusion_matrix, scores


def flatten_scores(matrix: torch.Tensor) -> dict[str, float]:
    result = {}
    for name, value in scores(matrix).items():
        if value.ndim == 0:
            result[name] = float(value)
        else:
            for index, score in enumerate(value):
                result[f"class_{index}_{name.removeprefix('per_class_')}"] = float(score)
    return result


class Validator:
    def __init__(self, *, model: torch.nn.Module, valid_loader: DataLoader,
                 criterion: torch.nn.Module, device: torch.device,
                 amp_dtype: torch.dtype, use_amp: bool,
                 preprocess: PreprocessSpec, predictor: Callable[..., np.ndarray],
                 ignore_index: int = 255) -> None:
        self.model = model
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_amp = use_amp
        self.preprocess = preprocess
        self.predictor = predictor
        self.ignore_index = ignore_index

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        totals = {}
        count = 0
        matrix = torch.zeros(self.model.num_classes, self.model.num_classes, dtype=torch.int64)
        for cpu_batch in self.valid_loader:
            batch = {k: v.to(self.device) for k, v in cpu_batch.items()}
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(batch["weak"])
                losses, _ = self.criterion(out, batch)
            logits = F.interpolate(out.logits[-1].float(), size=batch["mask"].shape[-2:],
                                   mode="bilinear", align_corners=False)
            matrix += confusion_matrix(logits, batch["mask"], batch["valid"],
                                       num_classes=self.model.num_classes,
                                       ignore_index=self.ignore_index).cpu()
            size = batch["weak"].shape[0]
            count += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
        if not count:
            raise RuntimeError("Validation loader is empty")
        return {**{k: v / count for k, v in totals.items()}, **flatten_scores(matrix)}

    def predict_native(self, loader):
        pairs = getattr(loader.dataset, "pairs", None) or getattr(loader.dataset, "data", None)
        if not pairs:
            raise RuntimeError("Deployment validation dataset has no image/mask pairs")
        self.model.eval()
        for image_path, mask_path in pairs:
            logits = self.predictor(self.model, read_image(image_path),
                                    size=self.preprocess.size, mode=self.preprocess.mode)
            yield logits.argmax(axis=0), read_mask(mask_path)

    def validate_deploy(self) -> dict[str, float]:
        matrix = torch.zeros(self.model.num_classes, self.model.num_classes, dtype=torch.int64)
        for pred, target in self.predict_native(self.valid_loader):
            matrix += confusion_matrix(torch.from_numpy(pred), torch.from_numpy(target),
                                       num_classes=self.model.num_classes,
                                       ignore_index=self.ignore_index).cpu()
        return {f"deploy_{k}": v for k, v in flatten_scores(matrix).items()}
