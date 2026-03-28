from typing import Any

from fastapi import Request

from app.core.error_codes import ErrorCode
from app.core.logging import get_request_id


def _resolve_request_id(
    *, request: Request | None = None, request_id: str | None = None
) -> str | None:
    if request_id:
        return request_id

    if request is not None:
        state_request_id = getattr(request.state, "request_id", None)
        if state_request_id:
            return state_request_id

    context_request_id = get_request_id()
    return None if context_request_id == "-" else context_request_id


def build_success_response(
    *,
    data: Any,
    request: Request | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "request_id": _resolve_request_id(request=request, request_id=request_id),
    }


def build_error_response(
    *,
    code: ErrorCode | str,
    message: str,
    details: Any | None = None,
    request: Request | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    normalized_code = code.value if isinstance(code, ErrorCode) else str(code)

    return {
        "success": False,
        "error": {
            "code": normalized_code,
            "message": message,
            "details": [] if details is None else details,
        },
        "request_id": _resolve_request_id(request=request, request_id=request_id),
    }
