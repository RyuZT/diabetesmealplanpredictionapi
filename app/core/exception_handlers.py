import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.constants import (
    DEFAULT_BAD_REQUEST_MESSAGE,
    DEFAULT_INTERNAL_ERROR_MESSAGE,
    DEFAULT_VALIDATION_MESSAGE,
)
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppException
from app.core.responses import build_error_response

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("Application exception: %s", exc.message)
        else:
            logger.warning("Application exception: %s", exc.message)

        payload = build_error_response(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request=request,
        )
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = []
        for item in exc.errors():
            location = [str(value) for value in item.get("loc", []) if value != "body"]
            details.append(
                {
                    "field": ".".join(location),
                    "message": item.get("msg", DEFAULT_VALIDATION_MESSAGE),
                    "type": item.get("type", "validation_error"),
                }
            )

        payload = build_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            message=DEFAULT_VALIDATION_MESSAGE,
            details=details,
            request=request,
        )
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if exc.status_code >= 500:
            code = ErrorCode.INTERNAL_SERVER_ERROR
            message = DEFAULT_INTERNAL_ERROR_MESSAGE
        elif exc.status_code == 400:
            code = ErrorCode.BAD_REQUEST
            message = exc.detail if isinstance(exc.detail, str) else DEFAULT_BAD_REQUEST_MESSAGE
        else:
            code = ErrorCode.HTTP_ERROR
            message = exc.detail if isinstance(exc.detail, str) else "request_failed"

        payload = build_error_response(
            code=code,
            message=message,
            request=request,
        )
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception", exc_info=exc)

        payload = build_error_response(
            code=ErrorCode.INTERNAL_SERVER_ERROR,
            message=DEFAULT_INTERNAL_ERROR_MESSAGE,
            request=request,
        )
        return JSONResponse(status_code=500, content=payload)
