from typing import Final

API_PREFIX: Final[str] = "/api/v1"

HEALTH_OK: Final[str] = "ok"
HEALTH_READY: Final[str] = "ready"
HEALTH_NOT_READY: Final[str] = "not_ready"

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

DEFAULT_VALIDATION_MESSAGE: Final[str] = "input is invalid"
DEFAULT_INTERNAL_ERROR_MESSAGE: Final[str] = "internal_server_error"
DEFAULT_BAD_REQUEST_MESSAGE: Final[str] = "bad_request"

MODEL_FEATURES: Final[tuple[str, ...]] = (
    "bmi",
    "age",
    "fgb",
    "avg_systolyc",
    "avg_dystolyc",
    "insulin_log",
)

NUMERIC_FOOD_COLUMNS: Final[tuple[str, ...]] = (
    "energy_kcal",
    "carbs",
    "protein_g",
    "fat_g",
    "freesugar_g",
    "fibre_g",
    "cholestrol_mg",
    "calcium_mg",
)

TOP_PREDICTIONS_LIMIT: Final[int] = 3
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.60

WARNING_MEAL_PLAN_UNAVAILABLE: Final[str] = "meal_plan_unavailable"
WARNING_NUTRITION_UNAVAILABLE: Final[str] = "nutrition_unavailable"
WARNING_LOW_CONFIDENCE: Final[str] = "low_confidence"
