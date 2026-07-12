"""Prometheus metrics used by the prediction API."""

from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total number of API requests.",
    ["method", "endpoint", "status_code"],
)

PREDICTION_COUNT = Counter(
    "heart_model_predictions_total",
    "Total number of model predictions.",
    ["prediction_class"],
)

PREDICTION_LATENCY = Histogram(
    "heart_model_prediction_seconds",
    "Time spent processing prediction requests.",
)

PREDICTION_ERRORS = Counter(
    "heart_model_prediction_errors_total",
    "Total number of model inference failures.",
)
