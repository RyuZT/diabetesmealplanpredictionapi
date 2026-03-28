from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router
from app.core.constants import API_PREFIX

api_router = APIRouter(prefix=API_PREFIX)
api_router.include_router(health_router)
api_router.include_router(predict_router)
