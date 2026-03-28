import logging

from fastapi import APIRouter, Depends, Request

from app.core.constants import HEALTH_OK, HEALTH_READY
from app.core.exceptions import ResourceNotReadyError
from app.core.responses import build_success_response
from app.core.settings import Settings, get_settings
from app.ml.model_loader import get_resource_readiness
from app.schemas.common import (
    HealthData,
    HealthResponse,
    ReadyData,
    ReadyResources,
    ReadyResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    payload = build_success_response(
        data=HealthData(
            status=HEALTH_OK,
            app_name=settings.app_name,
            app_version=settings.app_version,
        ).model_dump(),
        request=request,
    )
    return HealthResponse(**payload)


@router.get("/readyz", response_model=ReadyResponse)
def readyz(request: Request) -> ReadyResponse:
    readiness = get_resource_readiness()
    is_ready = all(readiness.values())

    logger.info("Ready check result: ready=%s", is_ready)

    if not is_ready:
        raise ResourceNotReadyError(
            message="required resources are not ready",
            details=readiness,
        )

    payload = build_success_response(
        data=ReadyData(
            status=HEALTH_READY,
            resources=ReadyResources(**readiness),
        ).model_dump(),
        request=request,
    )
    return ReadyResponse(**payload)
