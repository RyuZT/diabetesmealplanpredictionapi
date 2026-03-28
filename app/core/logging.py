from contextvars import ContextVar
import logging

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID_CTX.get()
        return True


def configure_logging(log_level: str = "INFO") -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root_logger.addHandler(handler)
        return

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        has_filter = any(isinstance(item, RequestIdFilter) for item in handler.filters)
        if not has_filter:
            handler.addFilter(RequestIdFilter())


def set_request_id(request_id: str) -> None:
    _REQUEST_ID_CTX.set(request_id)


def get_request_id() -> str:
    return _REQUEST_ID_CTX.get()


def clear_request_id() -> None:
    _REQUEST_ID_CTX.set("-")
