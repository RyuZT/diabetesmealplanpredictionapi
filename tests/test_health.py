from fastapi.testclient import TestClient

from app.api.routes import health as health_route
from app.main import app

client = TestClient(app)


def test_healthz_returns_ok() -> None:
    response = client.get("/api/v1/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_readyz_returns_ready_when_resources_available(monkeypatch) -> None:
    monkeypatch.setattr(
        health_route,
        "get_resource_readiness",
        lambda: {
            "model_loaded": True,
            "scaler_loaded": True,
            "label_encoder_loaded": True,
            "food_data_loaded": True,
        },
    )

    response = client.get("/api/v1/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ready"
    assert body["data"]["resources"]["model_loaded"] is True
    assert body["request_id"]


def test_readyz_returns_503_when_resources_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        health_route,
        "get_resource_readiness",
        lambda: {
            "model_loaded": False,
            "scaler_loaded": True,
            "label_encoder_loaded": True,
            "food_data_loaded": False,
        },
    )

    response = client.get("/api/v1/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "resource_not_ready"
    assert body["error"]["details"]["model_loaded"] is False
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]
