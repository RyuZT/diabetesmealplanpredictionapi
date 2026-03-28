from typing import Any

from app.core.error_codes import ErrorCode


class AppException(Exception):
    def __init__(
        self,
        *,
        code: ErrorCode,
        message: str,
        status_code: int,
        details: Any | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ModelNotLoadedError(AppException):
    def __init__(
        self,
        message: str = "model artifacts are not available",
        details: Any | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.MODEL_NOT_LOADED,
            message=message,
            status_code=500,
            details=details,
        )


class PredictionInputError(AppException):
    def __init__(
        self,
        message: str = "input is invalid",
        details: Any | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.BAD_REQUEST,
            message=message,
            status_code=400,
            details=details,
        )


class PredictionFailedError(AppException):
    def __init__(
        self,
        message: str = "prediction failed",
        details: Any | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.PREDICTION_FAILED,
            message=message,
            status_code=500,
            details=details,
        )


class ResourceNotReadyError(AppException):
    def __init__(
        self,
        message: str = "service is not ready",
        details: Any | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_READY,
            message=message,
            status_code=503,
            details=details,
        )
