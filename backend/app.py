import asyncio
from typing import Any

import torch
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.prepare.spec import PreprocessSpec
from .route import router
from .codec import MAX_BASE64_LENGTH

MAX_REQUEST_BYTES = MAX_BASE64_LENGTH + 64 * 1024


class RequestBodyLimit:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                await JSONResponse({"detail": "request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        pending = {"type": "http.request", "body": bytes(body), "more_body": False}
        del body

        async def replay() -> dict:
            nonlocal pending
            if pending is None:
                return await receive()
            message, pending = pending, None
            return message

        await self.app(scope, replay, send)


def read_preprocess(model: Any) -> PreprocessSpec:
    return PreprocessSpec.from_meta(getattr(model, "loaded_meta", None))


def build_app(
    model: Any,
    device: torch.device,
    preprocess: PreprocessSpec | None = None,
) -> FastAPI:
    app = FastAPI(title="BiRefNet Multiclass API")
    app.add_middleware(RequestBodyLimit)
    app.state.model = model
    app.state.device = device
    app.state.preprocess = read_preprocess(model) if preprocess is None else preprocess
    app.state.predict_sem = asyncio.Semaphore(1)
    app.include_router(router)
    return app
