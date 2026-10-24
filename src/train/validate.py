from collections.abc import Callable

import numpy as np
import torch
from ..data.image import read_image, read_mask
from ..data.pairs import Pair
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
    """Scores whole validation images at their own size, as prediction sees them."""

    def __init__(self, *, model: torch.nn.Module, pairs: list[Pair],
                 preprocess: PreprocessSpec, predictor: Callable[..., np.ndarray],
                 ignore_index: int = 255) -> None:
        if not pairs:
            raise ValueError("Validation needs at least one image/mask pair")
        self.model = model
        self.pairs = pairs
        self.preprocess = preprocess
        self.predictor = predictor
        self.ignore_index = ignore_index

    def validate(self) -> dict[str, float]:
        self.model.eval()
        matrix = torch.zeros(self.model.num_classes, self.model.num_classes, dtype=torch.int64)
        for image_path, mask_path in self.pairs:
            logits = self.predictor(self.model, read_image(image_path),
                                    size=self.preprocess.size, is_sem=self.preprocess.is_sem)
            matrix += confusion_matrix(torch.from_numpy(logits.argmax(axis=0)),
                                       torch.from_numpy(read_mask(mask_path)),
                                       num_classes=self.model.num_classes,
                                       ignore_index=self.ignore_index)
        return flatten_scores(matrix)
