"""Train, tune, evaluate, and track heart disease classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BEST_MODEL_PATH,
    CROSS_VALIDATION_FOLDS,
    MEDICAL_DISCLAIMER,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_DB_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_COMPARISON_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_REPORT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.data_processing import (  # noqa: E402
    build_preprocessor,
    clean_training_data,
    load_raw_data,
    prepare_features_and_target,
    save_processed_data,
)
from src.evaluation import (  # noqa: E402
    calculate_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_roc_curve,
    save_model_comparison_plot,
)


def create_logistic_search(
    cross_validation: StratifiedKFold,
) -> GridSearchCV:
    """Create Logistic Regression tuning workflow."""

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    parameter_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        "classifier__solver": [
            "liblinear",
            "lbfgs",
        ],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="roc_auc",
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=False,
    )


def create_random_forest_search(
    cross_validation: StratifiedKFold,
) -> RandomizedSearchCV:
    """Create Random Forest tuning workflow."""

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    parameter_distributions = {
        "classifier__n_estimators": [
            200,
            300,
            500,
        ],
        "classifier__max_depth": [
            None,
            8,
            12,
            16,
        ],
        "classifier__min_samples_split": [
            2,
            5,
            10,
        ],
        "classifier__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "classifier__max_features": [
            "sqrt",
            "log2",
            0.7,
        ],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
    }

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=15,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="roc_auc",
        cv=cross_validation,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False,
    )


def safe_parameter_value(value) -> str | int | float | bool:
    """Convert parameter values to MLflow-compatible values."""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def evaluate_and_log_model(
    model_name: str,
    search,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Tune, evaluate, and log one model to MLflow."""

    print(f"\nTraining {model_name}...")
    search.fit(x_train, y_train)

    best_pipeline = search.best_estimator_

    predictions = best_pipeline.predict(x_test)
    probabilities = best_pipeline.predict_proba(x_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    model_slug = model_name.lower().replace(" ", "_")
    model_artifact_dir = MODEL_REPORT_DIR / model_slug
    model_artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrix_path = model_artifact_dir / "confusion_matrix.png"
    roc_curve_path = model_artifact_dir / "roc_curve.png"
    report_path = model_artifact_dir / "classification_report.json"
    parameters_path = model_artifact_dir / "best_parameters.json"
    cv_results_path = model_artifact_dir / "cv_results.csv"

    save_confusion_matrix(
        y_test,
        predictions,
        confusion_matrix_path,
        model_name,
    )
    save_roc_curve(
        y_test,
        probabilities,
        roc_curve_path,
        model_name,
    )
    save_classification_report(
        y_test,
        predictions,
        report_path,
    )

    serializable_parameters = {
        key: safe_parameter_value(value) for key, value in search.best_params_.items()
    }

    parameters_path.write_text(
        json.dumps(
            serializable_parameters,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(search.cv_results_).to_csv(
        cv_results_path,
        index=False,
    )

    input_example = x_train.head(3).copy()

    signature = infer_signature(
        input_example,
        best_pipeline.predict(input_example),
    )

    with mlflow.start_run(run_name=model_slug):
        mlflow.set_tags(
            {
                "model_name": model_name,
                "dataset": "Combined UCI Heart Disease",
                "task": "binary_classification",
                "selection_metric": "cross_validation_roc_auc",
            }
        )

        mlflow.log_params(serializable_parameters)
        mlflow.log_param("training_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))
        mlflow.log_param(
            "cross_validation_folds",
            CROSS_VALIDATION_FOLDS,
        )
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric(
            "cv_best_roc_auc",
            float(search.best_score_),
        )

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(
                f"test_{metric_name}",
                float(metric_value),
            )

        mlflow.log_artifacts(
            str(model_artifact_dir),
            artifact_path="evaluation",
        )
        mlflow.log_artifact(
            str(PROJECT_ROOT / "reports" / "eda_summary.json"),
            artifact_path="dataset",
        )

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=input_example,
        )

    result = {
        "model_name": model_name,
        "cv_roc_auc": float(search.best_score_),
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "best_parameters": serializable_parameters,
        "pipeline": best_pipeline,
    }

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(
        "Test metrics: "
        + ", ".join(f"{name}={value:.4f}" for name, value in test_metrics.items())
    )

    return result


def save_best_model(
    results: list[dict],
    training_rows: int,
    test_rows: int,
) -> dict:
    """Select and serialize the model with highest CV ROC-AUC."""

    best_result = max(
        results,
        key=lambda result: result["cv_roc_auc"],
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_result["pipeline"],
        BEST_MODEL_PATH,
    )

    metadata = {
        "model_name": best_result["model_name"],
        "selection_reason": ("Highest five-fold stratified cross-validation ROC-AUC"),
        "cv_roc_auc": best_result["cv_roc_auc"],
        "test_accuracy": best_result["test_accuracy"],
        "test_precision": best_result["test_precision"],
        "test_recall": best_result["test_recall"],
        "test_f1": best_result["test_f1"],
        "test_roc_auc": best_result["test_roc_auc"],
        "training_rows": training_rows,
        "test_rows": test_rows,
        "random_state": RANDOM_STATE,
        "best_parameters": best_result["best_parameters"],
        "model_file": BEST_MODEL_PATH.name,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }

    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def save_model_comparison(results: list[dict]) -> None:
    """Save a concise comparison table without pipeline objects."""

    comparison_rows = []

    for result in results:
        comparison_rows.append(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "pipeline",
                    "best_parameters",
                }
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        by="cv_roc_auc",
        ascending=False,
    )

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    save_model_comparison_plot(
        comparison,
        MODEL_REPORT_DIR / "model_metric_comparison.png",
    )


def main() -> None:
    """Run complete model development workflow."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MLFLOW_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data = load_raw_data()
    cleaned_data = clean_training_data(raw_data)

    save_processed_data(cleaned_data)

    features, target = prepare_features_and_target(cleaned_data)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    cross_validation = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri(),
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    searches = [
        (
            "Logistic Regression",
            create_logistic_search(cross_validation),
        ),
        (
            "Random Forest",
            create_random_forest_search(cross_validation),
        ),
    ]

    results = []

    for model_name, search in searches:
        result = evaluate_and_log_model(
            model_name=model_name,
            search=search,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        results.append(result)

    save_model_comparison(results)

    metadata = save_best_model(
        results=results,
        training_rows=len(x_train),
        test_rows=len(x_test),
    )

    print("\nTraining workflow completed.")
    print(f"Cleaned records: {len(cleaned_data)}")
    print(f"Training records: {len(x_train)}")
    print(f"Test records: {len(x_test)}")
    print(f"Selected model: {metadata['model_name']}")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Comparison saved to: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
