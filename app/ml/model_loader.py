from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.constants import NUMERIC_FOOD_COLUMNS
from app.core.exceptions import ModelNotLoadedError
from app.core.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    scaler: Any
    label_encoder: Any


def _artifact_paths() -> dict[str, Path]:
    settings = get_settings()
    return {
        "model": settings.model_dir / settings.model_filename,
        "scaler": settings.model_dir / settings.scaler_filename,
        "label_encoder": settings.model_dir / settings.label_encoder_filename,
    }


@lru_cache
def get_model_artifacts() -> ModelArtifacts:
    paths = _artifact_paths()
    missing = [name for name, file_path in paths.items() if not file_path.exists()]
    if missing:
        raise ModelNotLoadedError(
            message="required model artifacts are missing",
            details={"missing": missing},
        )

    try:
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        label_encoder = joblib.load(paths["label_encoder"])
    except Exception as exc:
        raise ModelNotLoadedError("failed to load model artifacts") from exc

    logger.info("Model artifacts loaded")
    return ModelArtifacts(model=model, scaler=scaler, label_encoder=label_encoder)


@lru_cache
def get_food_data() -> pd.DataFrame:
    settings = get_settings()
    csv_path = settings.data_dir / settings.food_data_filename
    if not csv_path.exists():
        raise ModelNotLoadedError("food dataset is not available")

    try:
        df_food = pd.read_csv(csv_path, delimiter=";")
    except Exception as exc:
        raise ModelNotLoadedError("failed to load food dataset") from exc

    for col in NUMERIC_FOOD_COLUMNS:
        df_food[col] = pd.to_numeric(df_food[col], errors="coerce").fillna(0)

    if "fooditems" in df_food.columns:
        df_food["fooditems"] = df_food["fooditems"].fillna("").astype(str)

    logger.info("Food dataset loaded")
    return df_food


def get_resource_readiness() -> dict[str, bool]:
    readiness = {
        "model_loaded": False,
        "scaler_loaded": False,
        "label_encoder_loaded": False,
        "food_data_loaded": False,
    }

    try:
        artifacts = get_model_artifacts()
        readiness["model_loaded"] = artifacts.model is not None
        readiness["scaler_loaded"] = artifacts.scaler is not None
        readiness["label_encoder_loaded"] = artifacts.label_encoder is not None
    except Exception:
        logger.warning("Model artifacts are not ready")

    try:
        food_df = get_food_data()
        readiness["food_data_loaded"] = food_df is not None
    except Exception:
        logger.warning("Food dataset is not ready")

    return readiness
