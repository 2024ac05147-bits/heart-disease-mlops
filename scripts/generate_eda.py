"""Generate reproducible EDA artifacts for the combined UCI dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FIGURE_DIR,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    REPORT_DIR,
    TARGET_COLUMN,
)

from src.data_processing import (  # noqa: E402
    clean_training_data,
    load_raw_data,
)


def load_data() -> pd.DataFrame:
    """Load data and apply deterministic pre-modelling cleaning."""

    raw_data = load_raw_data()
    cleaned_data = clean_training_data(raw_data)

    required_context_columns = {
        "source_hospital",
        "original_target",
    }

    missing_context_columns = sorted(
        required_context_columns - set(cleaned_data.columns)
    )

    if missing_context_columns:
        raise ValueError(
            f"Dataset is missing EDA context columns: {missing_context_columns}"
        )

    return cleaned_data


def configure_plotting() -> None:
    """Configure consistent plotting defaults."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (9, 6)
    plt.rcParams["figure.dpi"] = 120


def save_class_distribution(data: pd.DataFrame) -> None:
    """Create target class distribution plot."""

    class_counts = (
        data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename(index={0: "No disease", 1: "Disease"})
    )

    axis = class_counts.plot(kind="bar")
    axis.set_title("Heart Disease Class Distribution")
    axis.set_xlabel("Target class")
    axis.set_ylabel("Number of records")
    axis.tick_params(axis="x", rotation=0)

    for container in axis.containers:
        axis.bar_label(container)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png")
    plt.close()


def save_source_distribution(data: pd.DataFrame) -> None:
    """Create record distribution plot by source hospital."""

    source_counts = data["source_hospital"].value_counts().sort_values(ascending=False)

    axis = source_counts.plot(kind="bar")
    axis.set_title("Patient Records by Source Hospital")
    axis.set_xlabel("Source hospital")
    axis.set_ylabel("Number of records")
    axis.tick_params(axis="x", rotation=20)

    for container in axis.containers:
        axis.bar_label(container)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "source_distribution.png")
    plt.close()


def save_target_rate_by_source(data: pd.DataFrame) -> None:
    """Create disease prevalence plot by source hospital."""

    prevalence = (
        data.groupby("source_hospital")[TARGET_COLUMN]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    axis = prevalence.plot(kind="bar")
    axis.set_title("Heart Disease Prevalence by Source Hospital")
    axis.set_xlabel("Source hospital")
    axis.set_ylabel("Heart disease positive records (%)")
    axis.tick_params(axis="x", rotation=20)

    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_rate_by_source.png")
    plt.close()


def save_missing_value_plot(data: pd.DataFrame) -> None:
    """Create missing-value count plot."""

    missing = data[MODEL_FEATURES].isna().sum().sort_values(ascending=False)

    axis = missing.plot(kind="bar")
    axis.set_title("Missing Values by Feature")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Missing-value count")
    axis.tick_params(axis="x", rotation=45)

    for container in axis.containers:
        axis.bar_label(container)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missing_values.png")
    plt.close()


def save_missing_value_percentage_plot(data: pd.DataFrame) -> None:
    """Create missing-value percentage plot."""

    missing_percentage = (
        data[MODEL_FEATURES].isna().mean().mul(100).sort_values(ascending=False)
    )

    axis = missing_percentage.plot(kind="bar")
    axis.set_title("Missing Data Percentage by Feature")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Missing values (%)")
    axis.tick_params(axis="x", rotation=45)

    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missing_value_percentage.png")
    plt.close()


def save_missingness_by_source(data: pd.DataFrame) -> None:
    """Create heatmap of missing-value percentage by source."""

    missing_by_source = data.groupby("source_hospital")[MODEL_FEATURES].apply(
        lambda frame: frame.isna().mean() * 100
    )

    plt.figure(figsize=(13, 5))
    sns.heatmap(
        missing_by_source,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Missing values (%)"},
    )

    plt.title("Missing Data Percentage by Source Hospital")
    plt.xlabel("Feature")
    plt.ylabel("Source hospital")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missingness_by_source.png")
    plt.close()


def save_histograms(data: pd.DataFrame) -> None:
    """Create numerical feature histograms."""

    numeric_data = data[NUMERICAL_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_data.hist(
        bins=20,
        figsize=(13, 9),
        edgecolor="black",
    )

    plt.suptitle(
        "Distribution of Numerical Health Features",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "numerical_histograms.png")
    plt.close()


def save_correlation_heatmap(data: pd.DataFrame) -> None:
    """Create correlation heatmap for numeric-compatible columns."""

    numeric_data = data[MODEL_FEATURES + [TARGET_COLUMN]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation = numeric_data.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png")
    plt.close()


def save_age_target_relationship(data: pd.DataFrame) -> None:
    """Show age distribution separated by target class."""

    plot_data = data.copy()
    plot_data["target_label"] = plot_data[TARGET_COLUMN].map(
        {0: "No disease", 1: "Disease"}
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=plot_data,
        x="age",
        hue="target_label",
        bins=20,
        multiple="layer",
        stat="count",
        alpha=0.55,
    )

    plt.title("Age Distribution by Heart Disease Outcome")
    plt.xlabel("Age")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "age_by_target.png")
    plt.close()


def save_chest_pain_target_relationship(
    data: pd.DataFrame,
) -> None:
    """Show disease rate by chest-pain category."""

    grouped = (
        data.groupby("cp", dropna=False)[TARGET_COLUMN].mean().mul(100).sort_index()
    )

    axis = grouped.plot(kind="bar")
    axis.set_title("Heart Disease Rate by Chest Pain Category")
    axis.set_xlabel("Chest pain category")
    axis.set_ylabel("Heart disease positive records (%)")
    axis.tick_params(axis="x", rotation=0)

    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "chest_pain_disease_rate.png")
    plt.close()


def save_feature_relationship_boxplots(
    data: pd.DataFrame,
) -> None:
    """Create key numerical feature boxplots by target."""
    FEATURE_LABELS = {
        "age": "Age",
        "thalach": "Maximum Heart Rate",
        "oldpeak": "ST Depression",
        "chol": "Serum Cholesterol",
    }

    plot_data = data.copy()
    plot_data["target_label"] = plot_data[TARGET_COLUMN].map(
        {0: "No disease", 1: "Disease"}
    )

    for feature in ["age", "thalach", "oldpeak", "chol"]:
        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=plot_data,
            x="target_label",
            y=feature,
        )

        label = FEATURE_LABELS[feature]
        plt.title(f"{label} Distribution by Heart Disease Outcome")
        plt.xlabel("Outcome")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{feature}_by_target_boxplot.png")
        plt.close()


def save_target_rate_by_sex(data: pd.DataFrame) -> None:
    """Create disease prevalence plot by sex category."""

    prevalence = data.groupby("sex")[TARGET_COLUMN].mean().mul(100).sort_index()

    prevalence.index = prevalence.index.map(
        {
            0: "Female",
            1: "Male",
        }
    )

    axis = prevalence.plot(kind="bar")
    axis.set_title("Heart Disease Prevalence by Sex")
    axis.set_xlabel("Sex")
    axis.set_ylabel("Heart disease positive records (%)")
    axis.tick_params(axis="x", rotation=0)

    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_rate_by_sex.png")
    plt.close()


def build_summary(data: pd.DataFrame) -> dict:
    """Build structured EDA summary."""

    target_counts = data[TARGET_COLUMN].value_counts().sort_index()

    missing_counts = data[MODEL_FEATURES].isna().sum()

    missing_percentages = data[MODEL_FEATURES].isna().mean().mul(100)

    numeric_summary = (
        data[MODEL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .describe()
        .round(3)
        .to_dict()
    )

    summary = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows_full_record": int(data.duplicated().sum()),
        "duplicate_feature_target_rows": int(
            data[MODEL_FEATURES + ["original_target"]].duplicated().sum()
        ),
        "class_counts": {
            str(key): int(value) for key, value in target_counts.to_dict().items()
        },
        "class_percentages": {
            str(key): round(float(value) * 100, 2)
            for key, value in data[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
            .items()
        },
        "positive_class_percentage": round(
            float(data[TARGET_COLUMN].mean()) * 100,
            2,
        ),
        "source_distribution": {
            str(key): int(value)
            for key, value in data["source_hospital"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "target_rate_by_source": {
            str(key): round(float(value) * 100, 2)
            for key, value in data.groupby("source_hospital")[TARGET_COLUMN]
            .mean()
            .sort_index()
            .to_dict()
            .items()
        },
        "missing_values": {
            column: int(value) for column, value in missing_counts.to_dict().items()
        },
        "missing_value_percentages": {
            column: round(float(value), 2)
            for column, value in missing_percentages.to_dict().items()
        },
        "numeric_summary": numeric_summary,
    }

    return summary


def save_eda_summary(data: pd.DataFrame) -> None:
    """Save EDA findings as JSON and readable text."""

    summary = build_summary(data)

    json_path = REPORT_DIR / "eda_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    text_lines = [
        "EDA SUMMARY",
        "===========",
        f"Rows: {summary['rows']}",
        f"Columns: {summary['columns']}",
        (f"Duplicate full rows: {summary['duplicate_rows_full_record']}"),
        (f"Duplicate feature-target rows: {summary['duplicate_feature_target_rows']}"),
        (f"Positive heart disease class: {summary['positive_class_percentage']}%"),
        "",
        "Class distribution:",
    ]

    for target, count in summary["class_counts"].items():
        percentage = summary["class_percentages"][target]
        text_lines.append(f"- Target {target}: {count} ({percentage}%)")

    text_lines.extend(
        [
            "",
            "Source distribution:",
        ]
    )

    for source, count in summary["source_distribution"].items():
        text_lines.append(f"- {source}: {count}")

    text_lines.extend(
        [
            "",
            "Heart disease rate by source:",
        ]
    )

    for source, rate in summary["target_rate_by_source"].items():
        text_lines.append(f"- {source}: {rate}%")

    text_lines.extend(
        [
            "",
            "Missing values:",
        ]
    )

    for column, count in summary["missing_values"].items():
        percentage = summary["missing_value_percentages"][column]

        text_lines.append(f"- {column}: {count} ({percentage}%)")

    text_path = REPORT_DIR / "eda_summary.txt"
    text_path.write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run all EDA generation tasks."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    data = load_data()

    save_class_distribution(data)
    save_source_distribution(data)
    save_target_rate_by_source(data)
    save_missing_value_plot(data)
    save_missing_value_percentage_plot(data)
    save_missingness_by_source(data)
    save_histograms(data)
    save_correlation_heatmap(data)
    save_age_target_relationship(data)
    save_chest_pain_target_relationship(data)
    save_feature_relationship_boxplots(data)
    save_target_rate_by_sex(data)
    save_eda_summary(data)

    print(f"EDA completed for {len(data)} records.")
    print(f"Figures saved under: {FIGURE_DIR}")
    print(f"Summary saved under: {REPORT_DIR}")


if __name__ == "__main__":
    main()
