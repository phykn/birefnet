import asyncio
from typing import Any

import torch
from fastapi import FastAPI

from src.prepare.spec import PreprocessSpec
from .route import router


def read_preprocess(model: Any) -> PreprocessSpec:
    return PreprocessSpec.from_meta(getattr(model, "loaded_meta", None))


def build_app(
    model: Any,
    device: torch.device,
    preprocess: PreprocessSpec | None = None,
) -> FastAPI:
    app = FastAPI(title="BiRefNet Multiclass API")
    app.state.model = model
    app.state.device = device
    app.state.preprocess = preprocess or PreprocessSpec()
    app.state.predict_sem = asyncio.Semaphore(1)
    app.include_router(router)
    return app
