"""Evaluation utilities for binary classification models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: pd.Series,
    predictions,
    probabilities,
) -> dict[str, float]:
    """Calculate held-out binary classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def save_confusion_matrix(
    y_true,
    predictions,
    output_path: Path,
    model_name: str,
) -> None:
    """Save confusion matrix plot."""

    matrix = confusion_matrix(y_true, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No disease",
            "Disease",
        ],
    )

    display.plot(values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_curve(
    y_true,
    probabilities,
    output_path: Path,
    model_name: str,
) -> None:
    """Save ROC curve plot."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        probabilities,
    )
    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_classification_report(
    y_true,
    predictions,
    output_path: Path,
) -> None:
    """Save classification report as JSON."""

    report = classification_report(
        y_true,
        predictions,
        target_names=[
            "No disease",
            "Disease",
        ],
        output_dict=True,
        zero_division=0,
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def save_model_comparison_plot(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save grouped comparison of model evaluation metrics."""

    metric_columns = [
        "cv_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]

    plot_data = comparison.set_index("model_name")[metric_columns].T

    axis = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_title("Classification Model Performance Comparison")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Model")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
