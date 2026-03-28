import logging
from datetime import datetime, timezone

import numpy as np

from app.core.constants import (
    LOW_CONFIDENCE_THRESHOLD,
    TOP_PREDICTIONS_LIMIT,
    WARNING_LOW_CONFIDENCE,
    WARNING_MEAL_PLAN_UNAVAILABLE,
    WARNING_NUTRITION_UNAVAILABLE,
)
from app.core.exceptions import PredictionFailedError, PredictionInputError
from app.core.settings import get_settings
from app.ml.model_loader import get_food_data, get_model_artifacts
from app.ml.postprocessing import parse_model_output
from app.ml.preprocessing import build_feature_dataframe
from app.schemas.predict import (
    PredictData,
    PredictRequest,
    PredictionMetadata,
    TopPrediction,
)
from app.services.meal_service import (
    build_nutrition_summary,
    normalize_status,
    recommend_meal_plan,
)

logger = logging.getLogger(__name__)


class InferenceService:
    def _build_probability_vector(self, model, features, label_encoder) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[0]

        predicted = model.predict(features)[0]
        class_count = len(label_encoder.classes_)
        probabilities = np.zeros(class_count, dtype=float)

        try:
            class_index = int(predicted)
        except (TypeError, ValueError):
            classes = [str(value) for value in label_encoder.classes_]
            class_index = classes.index(str(predicted)) if str(predicted) in classes else 0

        if class_index < 0 or class_index >= class_count:
            class_index = 0

        probabilities[class_index] = 1.0
        return probabilities

    def predict(self, payload: PredictRequest) -> PredictData:
        settings = get_settings()
        artifacts = get_model_artifacts()

        try:
            df_features = build_feature_dataframe(payload)
        except Exception as exc:
            raise PredictionInputError("input is invalid") from exc

        try:
            df_scaled = artifacts.scaler.transform(df_features)
            probabilities = self._build_probability_vector(
                artifacts.model,
                df_scaled,
                artifacts.label_encoder,
            )
            prediction, confidence, top_prediction_data = parse_model_output(
                probabilities,
                artifacts.label_encoder,
                top_k=TOP_PREDICTIONS_LIMIT,
            )
        except Exception as exc:
            logger.exception("Prediction failed during model inference")
            raise PredictionFailedError("prediction failed") from exc

        normalized_prediction = normalize_status(prediction)
        top_predictions = [TopPrediction(**item) for item in top_prediction_data]

        meal_plan = None
        nutrition = None
        warnings: list[str] = []

        try:
            food_df = get_food_data()
            meal_plan = recommend_meal_plan(food_df, normalized_prediction)
            if meal_plan is not None:
                nutrition = build_nutrition_summary(food_df, meal_plan)
        except Exception:
            logger.exception("Meal recommendation failed")

        if meal_plan is None:
            warnings.append(WARNING_MEAL_PLAN_UNAVAILABLE)

        if meal_plan is not None and nutrition is None:
            warnings.append(WARNING_NUTRITION_UNAVAILABLE)

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(WARNING_LOW_CONFIDENCE)

        logger.info("Prediction completed successfully")
        return PredictData(
            prediction=prediction,
            normalized_prediction=normalized_prediction,
            confidence=confidence,
            top_predictions=top_predictions,
            meal_plan=meal_plan,
            nutrition=nutrition,
            warnings=warnings,
            metadata=PredictionMetadata(
                model_version=settings.model_filename,
                inference_timestamp=datetime.now(timezone.utc),
            ),
        )


inference_service = InferenceService()
