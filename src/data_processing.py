"""Data cleaning and reusable preprocessing pipeline construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    INVALID_ZERO_AS_MISSING_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_raw_data() -> pd.DataFrame:
    """Load and validate the combined UCI Heart Disease dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. "
            "Run scripts/download_data.py first."
        )

    data = pd.read_csv(RAW_DATA_PATH)
    validate_model_data(data)

    return data


def validate_model_data(data: pd.DataFrame) -> None:
    """Validate schema, target values and feature availability."""

    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(data.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Dataset contains no records.")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    numeric_target = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    if numeric_target.isna().any():
        raise ValueError("Target contains non-numeric values.")

    invalid_targets = set(numeric_target.unique()) - {0, 1}

    if invalid_targets:
        raise ValueError(f"Unexpected target values found: {invalid_targets}")


def clean_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning before model splitting.

    The original UCI files sometimes use zero for unavailable blood
    pressure and cholesterol measurements. These values are converted
    to missing values so they can be imputed inside the model pipeline.
    """

    validate_model_data(data)
    cleaned = data.copy()

    for column in MODEL_FEATURES + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned[INVALID_ZERO_AS_MISSING_FEATURES] = cleaned[
        INVALID_ZERO_AS_MISSING_FEATURES
    ].replace(0, np.nan)

    if cleaned[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing or invalid values.")

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    invalid_targets = set(cleaned[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Unexpected target values found: {invalid_targets}")

    # Remove exact repeated feature-target rows to avoid duplicate records
    # appearing in separate train and test partitions.
    cleaned = cleaned.drop_duplicates(
        subset=MODEL_FEATURES + [TARGET_COLUMN],
        keep="first",
    ).reset_index(drop=True)

    return cleaned


def save_processed_data(data: pd.DataFrame) -> None:
    """Save cleaned data for reproducibility and inspection."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing used identically for training and inference."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=-1,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return modelling features and binary target."""

    features = data[MODEL_FEATURES].copy()
    target = data[TARGET_COLUMN].copy()

    return features, target
