"""Tests for deterministic cleaning and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES, TARGET_COLUMN
from src.data_processing import (
    build_preprocessor,
    clean_training_data,
    prepare_features_and_target,
    validate_model_data,
)


def make_test_data() -> pd.DataFrame:
    """Create representative raw records for unit testing."""

    return pd.DataFrame(
        [
            {
                "age": 57,
                "trestbps": 140,
                "chol": 241,
                "thalach": 123,
                "oldpeak": 0.2,
                "sex": 1,
                "cp": 4,
                "fbs": 0,
                "restecg": 1,
                "exang": 1,
                "slope": 2,
                "ca": 0,
                "thal": 7,
                "target": 1,
            },
            {
                "age": 45,
                "trestbps": 0,
                "chol": 0,
                "thalach": 165,
                "oldpeak": 0.0,
                "sex": 0,
                "cp": 2,
                "fbs": 0,
                "restecg": 0,
                "exang": 0,
                "slope": 1,
                "ca": np.nan,
                "thal": np.nan,
                "target": 0,
            },
        ]
    )


def test_validate_model_data_accepts_valid_schema() -> None:
    """Valid data should pass schema and target validation."""

    validate_model_data(make_test_data())


def test_validate_model_data_rejects_missing_feature() -> None:
    """Missing model features must be rejected."""

    data = make_test_data().drop(columns=["age"])

    try:
        validate_model_data(data)
    except ValueError as error:
        assert "missing required columns" in str(error).lower()
    else:
        raise AssertionError("Missing feature was not rejected.")


def test_validate_model_data_rejects_invalid_target() -> None:
    """The target must contain only binary values."""

    data = make_test_data()
    data.loc[0, TARGET_COLUMN] = 2

    try:
        validate_model_data(data)
    except ValueError as error:
        assert "unexpected target values" in str(error).lower()
    else:
        raise AssertionError("Invalid target was not rejected.")


def test_clean_training_data_converts_invalid_zero_values() -> None:
    """Zero blood pressure and cholesterol must become missing."""

    cleaned = clean_training_data(make_test_data())

    second_record = cleaned.iloc[1]

    assert pd.isna(second_record["trestbps"])
    assert pd.isna(second_record["chol"])


def test_clean_training_data_removes_duplicates() -> None:
    """Duplicate feature-target observations must be removed."""

    data = make_test_data()
    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    cleaned = clean_training_data(data)

    assert len(cleaned) == 2


def test_prepare_features_and_target() -> None:
    """Feature and target separation should preserve schema."""

    cleaned = clean_training_data(make_test_data())

    features, target = prepare_features_and_target(cleaned)

    assert list(features.columns) == MODEL_FEATURES
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target)


def test_preprocessor_fits_and_transforms_missing_data() -> None:
    """The transformer should impute, encode and scale successfully."""

    cleaned = clean_training_data(make_test_data())
    features, _ = prepare_features_and_target(cleaned)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(MODEL_FEATURES)
