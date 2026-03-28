from typing import Any

import numpy as np

from app.core.exceptions import PredictionFailedError


def decode_prediction_label(label_encoder: Any, class_index: int) -> str:
    try:
        return str(label_encoder.inverse_transform([class_index])[0])
    except Exception as exc:
        raise PredictionFailedError("prediction decoding failed") from exc


def parse_model_output(
    probabilities: np.ndarray,
    label_encoder: Any,
    *,
    top_k: int = 3,
) -> tuple[str, float, list[dict[str, float | str]]]:
    if probabilities.size == 0:
        raise PredictionFailedError("model did not return probabilities")

    sorted_indices = np.argsort(probabilities)[::-1]
    limit = max(1, min(top_k, int(probabilities.size)))

    top_predictions: list[dict[str, float | str]] = []
    for index in sorted_indices[:limit]:
        class_index = int(index)
        top_predictions.append(
            {
                "label": decode_prediction_label(label_encoder, class_index),
                "probability": float(probabilities[class_index]),
            }
        )

    best_prediction = top_predictions[0]
    return (
        str(best_prediction["label"]),
        float(best_prediction["probability"]),
        top_predictions,
    )
