"""FastAPI application for serving heart disease predictions."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    REQUEST_COUNT,
    PREDICTION_ERRORS,
)
from app.schemas import (
    HealthResponse,
    HeartDiseaseFeatures,
    PredictionResponse,
)

from src.config import (
    BEST_MODEL_PATH,
    MEDICAL_DISCLAIMER,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BEST_MODEL_PATH),
    )
)

METADATA_PATH = Path(
    os.getenv(
        "MODEL_METADATA_PATH",
        str(MODEL_METADATA_PATH),
    )
)

LOGGER = logging.getLogger("heart-disease-api")

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
)

MODEL = None
MODEL_METADATA: dict = {}


def load_model_resources() -> None:
    """Load the serialized model and associated metadata."""

    global MODEL
    global MODEL_METADATA

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at {MODEL_PATH}. "
            "Run scripts/train.py before starting the API."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata was not found at {METADATA_PATH}.")

    MODEL = joblib.load(MODEL_PATH)
    MODEL_METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    LOGGER.info(
        "Loaded model: %s",
        MODEL_METADATA.get("model_name", "unknown"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load application resources during service startup."""

    load_model_resources()
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description=(MEDICAL_DISCLAIMER),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
) -> Response:
    """Record request latency, status and structured logs."""

    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - started_at

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code="500",
        ).inc()

        LOGGER.exception(
            "method=%s endpoint=%s status=500 duration_seconds=%.4f",
            request.method,
            request.url.path,
            duration,
        )
        raise

    duration = time.perf_counter() - started_at

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).inc()

    LOGGER.info(
        "method=%s endpoint=%s status=%s duration_seconds=%.4f",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Report whether the API and model are ready."""

    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_name=MODEL_METADATA.get(
            "model_name",
            "unknown",
        ),
    )


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """Report whether the API process is running."""

    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> HealthResponse:
    """Report whether the prediction model is ready."""

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_name=MODEL_METADATA.get(
            "model_name",
            "unknown",
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    features: HeartDiseaseFeatures,
) -> PredictionResponse:
    """Predict heart disease presence and confidence."""

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not currently loaded.",
        )

    payload = features.model_dump()

    model_input = pd.DataFrame(
        [payload],
        columns=MODEL_FEATURES,
    ).astype("float64")

    try:
        with PREDICTION_LATENCY.time():
            prediction = int(MODEL.predict(model_input)[0])
            probabilities = MODEL.predict_proba(model_input)[0]
    except Exception as error:
        PREDICTION_ERRORS.inc()
        LOGGER.exception("Model inference failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from error

    probability_no_disease = float(probabilities[0])
    probability_disease = float(probabilities[1])
    confidence = max(
        probability_no_disease,
        probability_disease,
    )

    PREDICTION_COUNT.labels(prediction_class=str(prediction)).inc()

    LOGGER.info(
        "prediction=%s probability_disease=%.4f confidence=%.4f",
        prediction,
        probability_disease,
        confidence,
    )

    return PredictionResponse(
        prediction=prediction,
        risk_class=(
            "Disease risk detected" if prediction == 1 else "No disease risk detected"
        ),
        confidence=round(confidence, 4),
        probability_no_disease=round(
            probability_no_disease,
            4,
        ),
        probability_disease=round(
            probability_disease,
            4,
        ),
        model_name=MODEL_METADATA.get(
            "model_name",
            "unknown",
        ),
        disclaimer=MODEL_METADATA.get(
            "medical_disclaimer",
            "Academic demonstration only.",
        ),
    )


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus-compatible application metrics."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
