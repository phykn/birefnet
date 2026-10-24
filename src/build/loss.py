from typing import Any

from ..train.objective import TrainLoss


def build(cfg: Any) -> TrainLoss:
    return TrainLoss(
        lambda_cls=cfg.loss.ce,
        lambda_region=cfg.loss.dice,
        lambda_boundary=cfg.loss.boundary,
        boundary_radius=cfg.loss.boundary_width,
        lambda_aux=cfg.loss.aux,
        num_classes=cfg.model.classes,
        ignore_index=cfg.data.get("ignore_index", 255),
        label_blur=float(cfg.loss.get("label_blur", 0.0)),
    )
