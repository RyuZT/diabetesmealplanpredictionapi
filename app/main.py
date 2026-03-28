import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.constants import REQUEST_ID_HEADER
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import clear_request_id, configure_logging, set_request_id
from app.core.settings import get_settings
from app.ml.model_loader import get_resource_readiness

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        set_request_id(request_id)
        request.state.request_id = request_id

        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            if response is not None:
                response.headers[REQUEST_ID_HEADER] = request_id
            clear_request_id()

    app.include_router(api_router)
    register_exception_handlers(app)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Application startup completed")
        logger.info("Startup readiness: %s", get_resource_readiness())

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Application shutdown completed")

    return app


app = create_app()
