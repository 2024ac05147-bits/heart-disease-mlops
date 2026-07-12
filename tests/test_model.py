"""Tests for the packaged model pipeline."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
)


def test_model_artifact_exists() -> None:
    """The selected model must be packaged in the repository."""

    assert BEST_MODEL_PATH.exists()
    assert BEST_MODEL_PATH.stat().st_size > 0


def test_model_metadata_exists_and_is_valid() -> None:
    """Model metadata should describe the packaged estimator."""

    assert MODEL_METADATA_PATH.exists()

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["model_name"] in {
        "Logistic Regression",
        "Random Forest",
    }
    assert 0 <= metadata["test_roc_auc"] <= 1
    assert 0 <= metadata["test_recall"] <= 1
    assert metadata["model_file"] == BEST_MODEL_PATH.name
    assert metadata["medical_disclaimer"]


def test_packaged_pipeline_predicts(valid_payload: dict) -> None:
    """The serialized end-to-end pipeline must run inference."""

    model = joblib.load(BEST_MODEL_PATH)

    frame = pd.DataFrame(
        [valid_payload],
        columns=MODEL_FEATURES,
    ).astype("float64")

    prediction = model.predict(frame)
    probabilities = model.predict_proba(frame)

    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1}

    assert probabilities.shape == (1, 2)
    assert abs(float(probabilities[0].sum()) - 1.0) < 1e-8
