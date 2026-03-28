from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthData(BaseModel):
    status: str
    app_name: str
    app_version: str


class ReadyResources(BaseModel):
    model_loaded: bool
    scaler_loaded: bool
    label_encoder_loaded: bool
    food_data_loaded: bool


class ReadyData(BaseModel):
    status: str
    resources: ReadyResources


class HealthResponse(BaseModel):
    success: Literal[True] = True
    data: HealthData
    request_id: str | None = None


class ReadyResponse(BaseModel):
    success: Literal[True] = True
    data: ReadyData
    request_id: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = Field(default_factory=list)


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
    request_id: str | None = None
