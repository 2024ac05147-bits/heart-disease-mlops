"""Download, combine, validate, and standardize UCI Heart Disease data."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, RAW_DATA_PATH  # noqa: E402


UCI_DATASET_ID = 45

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

DATA_SOURCES = {
    "cleveland": f"{UCI_BASE_URL}/processed.cleveland.data",
    "hungarian": f"{UCI_BASE_URL}/processed.hungarian.data",
    "switzerland": f"{UCI_BASE_URL}/processed.switzerland.data",
    "va_long_beach": f"{UCI_BASE_URL}/processed.va.data",
}

EXPECTED_SOURCE_ROWS = {
    "cleveland": 303,
    "hungarian": 294,
    "switzerland": 123,
    "va_long_beach": 200,
}

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "original_target",
]

MODEL_FEATURES = FEATURE_COLUMNS[:-1]


def standardize_dataset(
    data: pd.DataFrame,
    source_hospital: str,
) -> pd.DataFrame:
    """Convert one hospital dataset to the common project schema."""

    if data.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{source_hospital} returned {data.shape[1]} columns; "
            f"expected {len(FEATURE_COLUMNS)}."
        )

    data = data.copy()
    data.columns = FEATURE_COLUMNS

    data = data.replace(
        {
            "?": pd.NA,
            "": pd.NA,
            " ": pd.NA,
        }
    )

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data["original_target"].isna().any():
        missing_targets = int(data["original_target"].isna().sum())
        raise ValueError(
            f"{source_hospital} contains {missing_targets} invalid target values."
        )

    invalid_original_targets = set(data["original_target"].dropna().unique()) - {
        0,
        1,
        2,
        3,
        4,
    }

    if invalid_original_targets:
        raise ValueError(
            f"{source_hospital} contains unexpected targets: "
            f"{sorted(invalid_original_targets)}"
        )

    # UCI target definition:
    # 0 = absence of heart disease
    # 1-4 = presence of heart disease
    data["target"] = (data["original_target"] > 0).astype(int)

    data["source_hospital"] = source_hospital
    data["uci_dataset_id"] = UCI_DATASET_ID

    return data


def download_source(
    source_hospital: str,
    url: str,
) -> pd.DataFrame:
    """Download one processed UCI hospital dataset."""

    print(f"Downloading {source_hospital}...")

    with urlopen(url, timeout=60) as response:
        raw_content = response.read()

    data = pd.read_csv(
        BytesIO(raw_content),
        header=None,
        na_values=["?"],
        skipinitialspace=True,
    )

    standardized = standardize_dataset(
        data,
        source_hospital=source_hospital,
    )

    expected_rows = EXPECTED_SOURCE_ROWS[source_hospital]

    if len(standardized) != expected_rows:
        print(
            f"Warning: {source_hospital} has "
            f"{len(standardized)} rows; expected approximately "
            f"{expected_rows}."
        )

    return standardized


def load_local_source(
    source_hospital: str,
    filename: str,
) -> pd.DataFrame:
    """Load a manually downloaded UCI source file."""

    local_path = RAW_DATA_DIR / filename

    if not local_path.exists():
        raise FileNotFoundError(f"Local fallback file not found: {local_path}")

    data = pd.read_csv(
        local_path,
        header=None,
        na_values=["?"],
        skipinitialspace=True,
    )

    return standardize_dataset(
        data,
        source_hospital=source_hospital,
    )


def download_all_processed_sources() -> pd.DataFrame:
    """Download and combine all four processed UCI datasets."""

    source_frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    for source_hospital, url in DATA_SOURCES.items():
        try:
            source_data = download_source(
                source_hospital,
                url,
            )
            source_frames.append(source_data)
        except (URLError, OSError, TimeoutError) as error:
            failures[source_hospital] = str(error)
            print(f"Automatic download failed for {source_hospital}: {error}")

    if failures:
        filenames = {
            "cleveland": "processed.cleveland.data",
            "hungarian": "processed.hungarian.data",
            "switzerland": "processed.switzerland.data",
            "va_long_beach": "processed.va.data",
        }

        downloaded_sources = {
            frame["source_hospital"].iloc[0] for frame in source_frames
        }

        for source_hospital in failures:
            if source_hospital in downloaded_sources:
                continue

            print(f"Trying local fallback for {source_hospital}...")

            source_frames.append(
                load_local_source(
                    source_hospital,
                    filenames[source_hospital],
                )
            )

    if len(source_frames) != len(DATA_SOURCES):
        loaded_sources = sorted(
            frame["source_hospital"].iloc[0] for frame in source_frames
        )
        raise RuntimeError(
            "Could not load all four UCI datasets. "
            f"Successfully loaded: {loaded_sources}"
        )

    combined = pd.concat(
        source_frames,
        ignore_index=True,
    )

    return combined


def try_ucimlrepo_for_metadata() -> dict[str, str]:
    """Retrieve optional UCI metadata without using it as training data."""

    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)

        return {
            "name": str(dataset.metadata.name),
            "repository_url": str(dataset.metadata.repository_url),
        }
    except (ConnectionError, URLError, OSError):
        return {
            "name": "Heart Disease",
            "repository_url": ("https://archive.ics.uci.edu/dataset/45/heart+disease"),
        }


def validate_combined_data(data: pd.DataFrame) -> dict:
    """Validate the combined dataset and create a summary."""

    if data.empty:
        raise ValueError("Combined dataset is empty.")

    expected_sources = set(DATA_SOURCES)
    actual_sources = set(data["source_hospital"].unique())

    missing_sources = expected_sources - actual_sources
    if missing_sources:
        raise ValueError(
            f"Combined dataset is missing sources: {sorted(missing_sources)}"
        )

    if data["target"].isna().any():
        raise ValueError("Binary target contains missing values.")

    invalid_targets = set(data["target"].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Unexpected binary targets: {invalid_targets}")

    source_counts = data["source_hospital"].value_counts().sort_index().to_dict()

    feature_duplicate_mask = data[MODEL_FEATURES + ["original_target"]].duplicated(
        keep=False
    )

    exact_duplicate_count = int(
        data[MODEL_FEATURES + ["original_target", "source_hospital"]].duplicated().sum()
    )

    metadata = try_ucimlrepo_for_metadata()

    summary = {
        "dataset_name": metadata["name"],
        "repository_url": metadata["repository_url"],
        "row_count_before_deduplication": int(len(data)),
        "column_count": int(len(data.columns)),
        "source_distribution": {
            key: int(value) for key, value in source_counts.items()
        },
        "exact_duplicates_within_source": exact_duplicate_count,
        "records_matching_across_any_source": int(feature_duplicate_mask.sum()),
        "target_distribution": {
            str(key): int(value)
            for key, value in data["target"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "target_percentage": {
            str(key): round(float(value) * 100, 2)
            for key, value in data["target"]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
            .items()
        },
        "missing_values": {
            column: int(count)
            for column, count in data[MODEL_FEATURES].isna().sum().items()
        },
    }

    return summary


def save_source_files(
    combined_data: pd.DataFrame,
) -> None:
    """Save standardized hospital-level CSV files for traceability."""

    standardized_dir = RAW_DATA_DIR / "standardized_sources"
    standardized_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source_hospital, source_data in combined_data.groupby("source_hospital"):
        output_path = standardized_dir / f"{source_hospital}.csv"
        source_data.to_csv(
            output_path,
            index=False,
        )


def main() -> None:
    """Download, combine, validate, and save all UCI sources."""

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading the four processed UCI Heart Disease datasets...")

    combined_data = download_all_processed_sources()

    summary = validate_combined_data(combined_data)

    save_source_files(combined_data)

    combined_data.to_csv(
        RAW_DATA_PATH,
        index=False,
    )

    summary_path = RAW_DATA_DIR / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nCombined dataset summary:")
    print(json.dumps(summary, indent=2))

    print(f"\nCombined dataset saved to: {RAW_DATA_PATH}")
    print(f"Dataset shape: {combined_data.shape}")


if __name__ == "__main__":
    main()
