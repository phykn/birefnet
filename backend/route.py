from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from src.predict.inference import predict as predict_mask

from .codec import ImageLimitError, decode, encode
from .schema import HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def check_health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", device=request.app.state.device.type)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    num_classes = request.app.state.model.num_classes
    if body.class_id is not None and body.class_id >= num_classes:
        raise HTTPException(status_code=422, detail="class_id must be in [0, num_classes)")

    async with request.app.state.predict_sem:
        try:
            image = await run_in_threadpool(decode, body.base64_str)
        except ImageLimitError as exc:
            raise HTTPException(status_code=413, detail="image too large") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc

        mask = await run_in_threadpool(
            predict_mask,
            request.app.state.model,
            image,
            output_mode=body.output_mode,
            class_id=body.class_id,
            size=request.app.state.preprocess.size,
            mode=request.app.state.preprocess.mode,
            tiles=body.tiles,
            overlap=body.overlap,
        )
        data = await run_in_threadpool(encode, mask, indexed=body.output_mode == "labels")

    height, width = mask.shape[:2]
    return PredictResponse(
        id=body.id,
        base64_str=data,
        height=height,
        width=width,
        channel=mask.shape[2] if mask.ndim == 3 else None,
        output_mode=body.output_mode,
        num_classes=num_classes,
        class_id=body.class_id,
        value_range=(0, num_classes - 1) if body.output_mode == "labels" else (0, 255),
    )
