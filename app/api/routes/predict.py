import logging

from fastapi import APIRouter, Request, status

from app.core.responses import build_success_response
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference_service import inference_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=False,
)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    logger.info("Prediction request received")
    prediction_data = inference_service.predict(payload)

    response_payload = build_success_response(
        data=prediction_data.model_dump(),
        request=request,
    )
    return PredictResponse(**response_payload)
