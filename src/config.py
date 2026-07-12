"""Central configuration for the heart disease MLOps project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "heart_disease_uci.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "heart_disease_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_REPORT_DIR = REPORT_DIR / "model_evaluation"
MODEL_COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CROSS_VALIDATION_FOLDS = 5

TARGET_COLUMN = "target"

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

INVALID_ZERO_AS_MISSING_FEATURES = [
    "trestbps",
    "chol",
]

MEDICAL_DISCLAIMER = (
    "This model is an academic demonstration and is not intended "
    "for clinical diagnosis or treatment decisions."
)
