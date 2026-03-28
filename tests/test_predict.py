from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.routes import predict as predict_route
from app.core.exceptions import PredictionFailedError
from app.main import app
from app.schemas.predict import (
    PredictData,
    PredictionMetadata,
    TopPrediction,
)

client = TestClient(app)


def valid_payload() -> dict:
    return {
        "bmi": 24.7,
        "age": 43,
        "fgb": 98,
        "avg_systolic": 122,
        "avg_diastolic": 80,
        "insulin": 12,
    }


def test_predict_success(monkeypatch) -> None:
    def fake_predict(_payload):
        return PredictData(
            prediction="Non-Diabetic",
            normalized_prediction="Non-Diabetic",
            confidence=0.91,
            top_predictions=[
                TopPrediction(label="Non-Diabetic", probability=0.91),
                TopPrediction(label="Prediabetic", probability=0.07),
            ],
            meal_plan=None,
            nutrition=None,
            warnings=["meal_plan_unavailable"],
            metadata=PredictionMetadata(
                model_version="best_xgb.pkl",
                inference_timestamp=datetime(2026, 3, 28, 8, 10, 31, tzinfo=timezone.utc),
            ),
        )

    monkeypatch.setattr(predict_route.inference_service, "predict", fake_predict)

    response = client.post(
        "/api/v1/predict",
        json=valid_payload(),
        headers={"X-Request-ID": "req-123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["request_id"] == "req-123"
    assert body["data"]["prediction"] == "Non-Diabetic"
    assert body["data"]["top_predictions"][0]["label"] == "Non-Diabetic"
    assert "warnings" in body["data"]
    assert body["data"]["warnings"] == ["meal_plan_unavailable"]
    assert body["data"]["metadata"]["model_version"] == "best_xgb.pkl"
    assert body["data"]["metadata"]["inference_timestamp"]
    assert "meal_plan" in body["data"]
    assert body["data"]["meal_plan"] is None
    assert "nutrition" in body["data"]
    assert body["data"]["nutrition"] is None
    assert response.headers["X-Request-ID"] == "req-123"


def test_predict_validation_error_shape() -> None:
    payload = valid_payload()
    payload["bmi"] = -1

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["details"], list)
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_predict_app_exception_shape(monkeypatch) -> None:
    def fake_predict(_payload):
        raise PredictionFailedError("prediction failed")

    monkeypatch.setattr(predict_route.inference_service, "predict", fake_predict)

    response = client.post("/api/v1/predict", json=valid_payload())

    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "prediction_failed"
    assert body["error"]["message"] == "prediction failed"
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]
