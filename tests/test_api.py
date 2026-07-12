"""Integration tests for the FastAPI prediction service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint() -> None:
    """Root endpoint should describe the service."""

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == "Heart Disease Risk Prediction API"
    assert body["documentation"] == "/docs"
    assert body["health"] == "/health"
    assert body["metrics"] == "/metrics"


def test_liveness_endpoint() -> None:
    """Liveness endpoint should confirm the process is running."""

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint() -> None:
    """Readiness endpoint should confirm the model is loaded."""

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["model_name"] in {
        "Logistic Regression",
        "Random Forest",
    }

def test_prediction_endpoint(valid_payload: dict) -> None:
    """A valid JSON request should return prediction probabilities."""

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=valid_payload,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in {0, 1}
    assert body["risk_class"] in {
        "Disease risk detected",
        "No disease risk detected",
    }
    assert 0 <= body["confidence"] <= 1
    assert 0 <= body["probability_no_disease"] <= 1
    assert 0 <= body["probability_disease"] <= 1

    probability_total = body["probability_no_disease"] + body["probability_disease"]

    assert abs(probability_total - 1.0) < 0.001
    assert body["model_name"] in {
        "Logistic Regression",
        "Random Forest",
    }
    assert body["disclaimer"]


def test_prediction_rejects_invalid_age(
    valid_payload: dict,
) -> None:
    """Invalid clinical ranges should return validation errors."""

    invalid_payload = valid_payload.copy()
    invalid_payload["age"] = 5

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_payload,
        )

    assert response.status_code == 422


def test_prediction_rejects_unknown_field(
    valid_payload: dict,
) -> None:
    """Unexpected JSON fields should not be silently accepted."""

    invalid_payload = valid_payload.copy()
    invalid_payload["unknown_feature"] = 123

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_payload,
        )

    assert response.status_code == 422


def test_prediction_accepts_optional_missing_values(
    valid_payload: dict,
) -> None:
    """Optional clinical fields should support pipeline imputation."""

    payload = valid_payload.copy()
    payload["ca"] = None
    payload["thal"] = None
    payload["slope"] = None

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 200
    assert response.json()["prediction"] in {0, 1}


def test_metrics_endpoint() -> None:
    """Prometheus endpoint should expose custom API metrics."""

    with TestClient(app) as client:
        client.get("/health/live")
        response = client.get("/metrics")

    assert response.status_code == 200

    content = response.text

    assert "heart_api_requests_total" in content
    assert "heart_model_predictions_total" in content
    assert "heart_model_prediction_seconds" in content
    assert "heart_model_prediction_errors_total" in content
