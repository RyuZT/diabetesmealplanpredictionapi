from enum import StrEnum


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    MODEL_NOT_LOADED = "model_not_loaded"
    PREDICTION_FAILED = "prediction_failed"
    RESOURCE_NOT_READY = "resource_not_ready"
    BAD_REQUEST = "bad_request"
    HTTP_ERROR = "http_error"
