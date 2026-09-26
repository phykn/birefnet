from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from .codec import MAX_BASE64_LENGTH


class HealthResponse(BaseModel):
    status: str
    device: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    base64_str: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH)
    output_mode: Literal["labels", "probability"] = "labels"
    class_id: int | None = Field(None, ge=0, strict=True)
    tiles: tuple[PositiveInt, ...] = Field((1,), min_length=1)
    overlap: float = Field(1 / 3, ge=1 / 3, lt=1.0)

    @model_validator(mode="after")
    def validate_class_id(self):
        if self.output_mode == "probability" and self.class_id is None:
            raise ValueError("probability output requires class_id")
        if self.output_mode == "labels" and self.class_id is not None:
            raise ValueError("class_id is only supported for probability output")
        return self


class PredictResponse(BaseModel):
    id: str | None
    base64_str: str
    height: int
    width: int
    channel: int | None
    output_mode: Literal["labels", "probability"]
    num_classes: int
    class_id: int | None
    dtype: Literal["uint8"] = "uint8"
    value_range: tuple[int, int]
