"""Evaluation pipeline for synthetic logical-reasoning tasks."""

from __future__ import annotations

import csv
import hashlib
import weakref
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.logical_dataset import (
    LOGICAL_TASKS,
    N_SAMPLES,
    generate_logical_dataset,
    primitive_conditions,
    recompute_target,
)
from metrics import evaluate_classifier
from models import get_model


EXPERIMENT_NAME = "logical_reasoning"
LOGICAL_LEVELS = ("L1", "L2", "L3")
N_TRAIN = 750
N_TEST = 250
CLASS_RATE_TOLERANCE = 0.10
ALL_MODELS = {
    "catboost",
    "realmlp",
    "tabicl_v2",
    "limix",
    "tabpfn_v2",
    "tabpfn_v3",
}
FULL_SEEDS = {0, 1, 2, 3, 4}
FULL_NOISE_SETTINGS = {0, 4}

OUTPUT_DIR = Path("results") / EXPERIMENT_NAME
PLOTS_DIR = OUTPUT_DIR / "plots"
RAW_RESULTS_PATH = OUTPUT_DIR / "logical_raw_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "logical_summary.csv"
VALIDATION_PATH = OUTPUT_DIR / "logical_validation.csv"
NO_NOISE_PLOT_PATH = PLOTS_DIR / "logical_without_irrelevant_features.png"
WITH_NOISE_PLOT_PATH = PLOTS_DIR / "logical_with_irrelevant_features.png"

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "fit_seconds",
    "prediction_seconds",
    "total_seconds",
    "log_loss",
    "roc_auc",
]
RAW_COLUMNS = [
    "dataset_module",
    "dataset_variant",
    "logical_operator",
    "n_noise_features",
    "model",
    "seed",
    "n_samples",
    "n_relevant_features",
    "n_features",
    "train_samples",
    "test_samples",
    "positive_rate",
    *METRIC_COLUMNS,
    "status",
    "error",
]
VALIDATION_COLUMNS = [
    "dataset_module",
    "dataset_variant",
    "logical_operator",
    "n_noise_features",
    "model",
    "seed",
    "dataset_shape",
    "n_relevant_features",
    "n_noise_features_observed",
    "n_total_features",
    "positive_rate",
    "negative_rate",
    "expected_positive_rate",
    "class_rate_close_to_expected",
    "class_0_count",
    "class_1_count",
    "primitive_conditions_recomputed",
    "label_rule_valid",
    "labels_binary",
    "n_samples_valid",
    "n_relevant_features_valid",
    "n_noise_features_valid",
    "n_total_features_valid",
    "finite_values_valid",
    "engineered_variables_absent",
    "train_indices_hash",
    "test_indices_hash",
    "same_split_for_all_models",
    "fresh_model_instance",
    "predictions_newly_generated",
    "validation_passed",
]


class PredictionTrackingClassifier:
    """Delegate to a model and record new prediction calls."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.predict_calls = 0
        self.predict_proba_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PredictionTrackingClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Any:
        self.predict_calls += 1
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Any:
        self.predict_proba_calls += 1
        return self.model.predict_proba(X)


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_dataset_and_split(
    X: np.ndarray,
    y: np.ndarray,
    dataset_variant: str,
    n_noise_features: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    """Validate logical labels, features, class rates, and shared indices."""
    task = LOGICAL_TASKS[dataset_variant]
    n_relevant = int(task["n_relevant_features"])
    conditions = primitive_conditions(X[:, :n_relevant], dataset_variant)
    direct_target = recompute_target(X[:, :n_relevant], dataset_variant)
    values, counts = np.unique(y, return_counts=True)
    class_counts = {int(value): int(count) for value, count in zip(values, counts)}
    positive_rate = float(np.mean(y))
    negative_rate = float(1.0 - positive_rate)
    expected_rate = float(task["expected_rate"])

    validation = {
        "dataset_shape": f"{X.shape[0]}x{X.shape[1]}",
        "n_relevant_features": n_relevant,
        "n_noise_features_observed": n_noise_features,
        "n_total_features": int(X.shape[1]),
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "expected_positive_rate": expected_rate,
        "class_rate_close_to_expected": bool(
            abs(positive_rate - expected_rate) <= CLASS_RATE_TOLERANCE
        ),
        "class_0_count": class_counts.get(0, 0),
        "class_1_count": class_counts.get(1, 0),
        "primitive_conditions_recomputed": bool(
            conditions.shape == (N_SAMPLES, n_relevant)
        ),
        "label_rule_valid": bool(np.array_equal(y, direct_target)),
        "labels_binary": bool(set(values.tolist()).issubset({0, 1})),
        "n_samples_valid": bool(X.shape[0] == N_SAMPLES),
        "n_relevant_features_valid": bool(
            n_relevant == int(task["n_relevant_features"])
        ),
        "n_noise_features_valid": bool(n_noise_features in FULL_NOISE_SETTINGS),
        "n_total_features_valid": bool(
            X.shape[1] == n_relevant + n_noise_features
        ),
        "finite_values_valid": bool(
            np.isfinite(X).all() and np.isfinite(y).all()
        ),
        # The dataset returns only original relevant/noise columns. The
        # primitive condition matrix is recomputed separately and never joined
        # to X.
        "engineered_variables_absent": bool(
            X.shape[1] == n_relevant + n_noise_features
        ),
        "train_indices_hash": _indices_hash(train_indices),
        "test_indices_hash": _indices_hash(test_indices),
    }
    required_flags = [
        "class_rate_close_to_expected",
        "primitive_conditions_recomputed",
        "label_rule_valid",
        "labels_binary",
        "n_samples_valid",
        "n_relevant_features_valid",
        "n_noise_features_valid",
        "n_total_features_valid",
        "finite_values_valid",
        "engineered_variables_absent",
    ]
    passed = bool(
        all(validation[flag] for flag in required_flags)
        and len(train_indices) == N_TRAIN
        and len(test_indices) == N_TEST
        and len(np.intersect1d(train_indices, test_indices)) == 0
    )
    if not passed:
        raise AssertionError(
            f"Logical dataset validation failed for {dataset_variant}, "
            f"noise={n_noise_features}: {validation}"
        )
    return validation


def _create_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    successful = raw_results.loc[raw_results["status"] == "success"].copy()
    grouped = successful.groupby(
        [
            "model",
            "dataset_variant",
            "logical_operator",
            "n_noise_features",
        ],
        sort=True,
    )
    summary = grouped[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    counts = (
        grouped["seed"]
        .nunique()
        .rename("n_successful_seeds")
        .reset_index()
    )
    return counts.merge(
        summary,
        on=[
            "model",
            "dataset_variant",
            "logical_operator",
            "n_noise_features",
        ],
        how="left",
    )


def _create_balanced_accuracy_plot(
    summary: pd.DataFrame,
    n_noise_features: int,
    title: str,
    path: Path,
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    level_positions = {level: index for index, level in enumerate(LOGICAL_LEVELS)}
    rows_for_noise = summary.loc[
        summary["n_noise_features"] == n_noise_features
    ]
    for model_name in sorted(rows_for_noise["model"].unique()):
        rows = rows_for_noise.loc[
            rows_for_noise["model"] == model_name
        ].copy()
        rows["level_position"] = rows["dataset_variant"].map(level_positions)
        rows = rows.sort_values("level_position")
        axis.errorbar(
            rows["level_position"],
            rows["balanced_accuracy_mean"],
            yerr=rows["balanced_accuracy_std"].fillna(0.0),
            marker="o",
            markersize=5,
            capsize=3,
            linewidth=1.5,
            label=model_name,
        )
    axis.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Chance",
    )
    axis.set_xticks(range(len(LOGICAL_LEVELS)), LOGICAL_LEVELS)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Logical task")
    axis.set_ylabel("Mean balanced accuracy")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _is_full_benchmark(raw_results: pd.DataFrame) -> bool:
    expected = {
        (model, level, noise, seed)
        for model in ALL_MODELS
        for level in LOGICAL_LEVELS
        for noise in FULL_NOISE_SETTINGS
        for seed in FULL_SEEDS
    }
    actual = set(
        map(
            tuple,
            raw_results.loc[
                raw_results["status"] == "success",
                ["model", "dataset_variant", "n_noise_features", "seed"],
            ].itertuples(index=False, name=None),
        )
    )
    return expected.issubset(actual)


def run_logical_benchmark(
    model_names: list[str],
    seeds: list[int],
    noise_settings: list[int],
) -> dict[str, Path]:
    """Run validated logical tasks using one shared split per condition."""
    invalid_noise = set(noise_settings) - FULL_NOISE_SETTINGS
    if invalid_noise:
        raise ValueError(
            f"Logical noise settings must be 0 or 4, got {invalid_noise}."
        )

    splits: dict[tuple[str, int, int], dict[str, Any]] = {}
    for n_noise_features in noise_settings:
        for dataset_variant in LOGICAL_LEVELS:
            for seed in seeds:
                X, y = generate_logical_dataset(
                    dataset_variant=dataset_variant,
                    n_noise_features=n_noise_features,
                    seed=seed,
                )
                indices = np.arange(N_SAMPLES)
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=0.25,
                    random_state=seed,
                    stratify=y,
                )
                validation = validate_dataset_and_split(
                    X,
                    y,
                    dataset_variant,
                    n_noise_features,
                    train_indices,
                    test_indices,
                )
                splits[(dataset_variant, n_noise_features, seed)] = {
                    "X_train": X[train_indices],
                    "X_test": X[test_indices],
                    "y_train": y[train_indices],
                    "y_test": y[test_indices],
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "validation": validation,
                }
                print(
                    "Validation | "
                    f"variant={dataset_variant} | "
                    f"shape={validation['dataset_shape']} | "
                    f"relevant={validation['n_relevant_features']} | "
                    f"noise={n_noise_features} | "
                    f"class_distribution="
                    f"{{0: {validation['class_0_count']}, "
                    f"1: {validation['class_1_count']}}} | "
                    f"positive_rate={validation['positive_rate']:.3f} | "
                    f"label_recomputed={validation['label_rule_valid']}"
                )

    model_factories = get_model(model_names)
    raw_rows = (
        pd.read_csv(RAW_RESULTS_PATH).to_dict("records")
        if RAW_RESULTS_PATH.exists()
        else []
    )
    validation_rows = (
        pd.read_csv(VALIDATION_PATH).to_dict("records")
        if VALIDATION_PATH.exists()
        else []
    )
    completed = {
        (
            str(row["model"]),
            str(row["dataset_variant"]),
            int(row["n_noise_features"]),
            int(row["seed"]),
        )
        for row in raw_rows
    }
    seen_model_references: list[weakref.ReferenceType[Any]] = []

    for n_noise_features in noise_settings:
        for dataset_variant in LOGICAL_LEVELS:
            for seed in seeds:
                split = splits[(dataset_variant, n_noise_features, seed)]
                expected_train_hash = _indices_hash(split["train_indices"])
                expected_test_hash = _indices_hash(split["test_indices"])
                for model_name, model_factory in model_factories.items():
                    run_key = (
                        model_name,
                        dataset_variant,
                        n_noise_features,
                        seed,
                    )
                    if run_key in completed:
                        print(f"Skipping completed run: {run_key}")
                        continue

                    same_split = bool(
                        _indices_hash(split["train_indices"])
                        == expected_train_hash
                        and _indices_hash(split["test_indices"])
                        == expected_test_hash
                    )
                    if not same_split:
                        raise AssertionError("Shared split indices changed.")

                    task = LOGICAL_TASKS[dataset_variant]
                    print(
                        f"Running {model_name} | variant={dataset_variant} | "
                        f"noise={n_noise_features} | seed={seed}"
                    )
                    base_row = {
                        "dataset_module": "logical_dataset",
                        "dataset_variant": dataset_variant,
                        "logical_operator": task["operator"],
                        "n_noise_features": n_noise_features,
                        "model": model_name,
                        "seed": seed,
                        "n_samples": N_SAMPLES,
                        "n_relevant_features": int(
                            task["n_relevant_features"]
                        ),
                        "n_features": int(split["X_train"].shape[1]),
                        "train_samples": N_TRAIN,
                        "test_samples": N_TEST,
                        "positive_rate": split["validation"]["positive_rate"],
                    }
                    fresh_model = False
                    predictions_newly_generated = False
                    try:
                        underlying_model = model_factory()
                        fresh_model = not any(
                            reference() is underlying_model
                            for reference in seen_model_references
                        )
                        if not fresh_model:
                            raise AssertionError("Model instance was reused.")
                        seen_model_references = [
                            reference
                            for reference in seen_model_references
                            if reference() is not None
                        ]
                        seen_model_references.append(
                            weakref.ref(underlying_model)
                        )
                        tracked_model = PredictionTrackingClassifier(
                            underlying_model
                        )
                        metrics = evaluate_classifier(
                            tracked_model,
                            split["X_train"],
                            split["X_test"],
                            split["y_train"],
                            split["y_test"],
                        )
                        predictions_newly_generated = (
                            tracked_model.predict_calls >= 1
                        )
                        if not predictions_newly_generated:
                            raise AssertionError(
                                "No new prediction call was observed."
                            )
                        row = {
                            **base_row,
                            **metrics,
                            "status": "success",
                            "error": "",
                        }
                        print(
                            f"Smoke/result accuracy={metrics['accuracy']:.4f} | "
                            "balanced_accuracy="
                            f"{metrics['balanced_accuracy']:.4f}"
                        )
                    except Exception as error:
                        row = {
                            **base_row,
                            **{metric: np.nan for metric in METRIC_COLUMNS},
                            "status": "failed",
                            "error": str(error),
                        }
                        print(f"Failed: {error}")

                    dataset_validation = split["validation"]
                    run_validation = {
                        "dataset_module": "logical_dataset",
                        "dataset_variant": dataset_variant,
                        "logical_operator": task["operator"],
                        "n_noise_features": n_noise_features,
                        "model": model_name,
                        "seed": seed,
                        **dataset_validation,
                        "same_split_for_all_models": same_split,
                        "fresh_model_instance": fresh_model,
                        "predictions_newly_generated": predictions_newly_generated,
                    }
                    required_flags = [
                        "class_rate_close_to_expected",
                        "primitive_conditions_recomputed",
                        "label_rule_valid",
                        "labels_binary",
                        "n_samples_valid",
                        "n_relevant_features_valid",
                        "n_noise_features_valid",
                        "n_total_features_valid",
                        "finite_values_valid",
                        "engineered_variables_absent",
                    ]
                    run_validation["validation_passed"] = bool(
                        all(
                            dataset_validation[flag]
                            for flag in required_flags
                        )
                        and same_split
                        and fresh_model
                        and predictions_newly_generated
                    )
                    raw_rows.append(row)
                    validation_rows.append(run_validation)
                    completed.add(run_key)
                    _write_rows(RAW_RESULTS_PATH, raw_rows, RAW_COLUMNS)
                    _write_rows(
                        VALIDATION_PATH,
                        validation_rows,
                        VALIDATION_COLUMNS,
                    )

    raw_results = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)
    summary = _create_summary(raw_results)
    summary.to_csv(SUMMARY_PATH, index=False)

    artifacts = {
        "raw_results": RAW_RESULTS_PATH,
        "summary": SUMMARY_PATH,
        "validation": VALIDATION_PATH,
    }
    if _is_full_benchmark(raw_results):
        _create_balanced_accuracy_plot(
            summary,
            0,
            "Model performance on logical-reasoning tasks without irrelevant features",
            NO_NOISE_PLOT_PATH,
        )
        _create_balanced_accuracy_plot(
            summary,
            4,
            "Model performance on logical-reasoning tasks with irrelevant features",
            WITH_NOISE_PLOT_PATH,
        )
        artifacts["without_noise_plot"] = NO_NOISE_PLOT_PATH
        artifacts["with_noise_plot"] = WITH_NOISE_PLOT_PATH
    else:
        print(
            "Report plots deferred until all six models, five seeds, "
            "three logical tasks, and both noise settings are complete."
        )
    return artifacts
