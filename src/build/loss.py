from typing import Any

from ..train.objective import TrainLoss


def build(cfg: Any) -> TrainLoss:
    return TrainLoss(
        lambda_cls=cfg.loss.lambda_cls,
        lambda_region=cfg.loss.lambda_region,
        lambda_boundary=cfg.loss.lambda_boundary,
        boundary_radius=cfg.loss.boundary_radius,
        lambda_aux=cfg.loss.lambda_aux,
        num_classes=cfg.birefnet.num_classes,
        ignore_index=cfg.data.get("ignore_index", 255),
    )
