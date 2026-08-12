"""L3-only irrelevant-feature robustness sweep for logical reasoning."""

from __future__ import annotations

import argparse
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
from logical_evaluation import (
    METRIC_COLUMNS,
    RAW_COLUMNS,
    VALIDATION_COLUMNS,
    PredictionTrackingClassifier,
    _indices_hash,
    _write_rows,
)
from metrics import evaluate_classifier
from models import get_model


EXPERIMENT_NAME = "logical_l3_noise_sweep"
DATASET_VARIANT = "L3"
LOGICAL_OPERATOR = "XOR"
NOISE_LEVELS = (0, 2, 4, 8)
REQUIRED_SEEDS = (0, 1, 2, 3, 4)
REQUIRED_MODELS = {
    "catboost",
    "realmlp",
    "tabicl_v2",
    "limix",
    "tabpfn_v2",
    "tabpfn_v3",
}
N_RELEVANT_FEATURES = 4
N_TRAIN = 750
N_TEST = 250
CLASS_RATE_TOLERANCE = 0.10

OUTPUT_DIR = Path("results") / EXPERIMENT_NAME
PLOTS_DIR = OUTPUT_DIR / "plots"
RAW_RESULTS_PATH = OUTPUT_DIR / "logical_l3_noise_sweep_raw.csv"
SUMMARY_PATH = OUTPUT_DIR / "logical_l3_noise_sweep_summary.csv"
VALIDATION_PATH = OUTPUT_DIR / "logical_l3_noise_sweep_validation.csv"
PLOT_PATH = PLOTS_DIR / "logical_l3_noise_sweep_balanced_accuracy.png"

SUMMARY_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "f1_macro",
    "roc_auc",
    "log_loss",
    "total_seconds",
]


def validate_l3_dataset_and_split(
    X: np.ndarray,
    y: np.ndarray,
    n_noise_features: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    """Validate the unchanged L3 XOR rule and sweep-specific dimensions."""
    conditions = primitive_conditions(
        X[:, :N_RELEVANT_FEATURES],
        DATASET_VARIANT,
    )
    direct_target = recompute_target(
        X[:, :N_RELEVANT_FEATURES],
        DATASET_VARIANT,
    )
    values, counts = np.unique(y, return_counts=True)
    class_counts = {
        int(value): int(count) for value, count in zip(values, counts)
    }
    positive_rate = float(np.mean(y))
    validation = {
        "dataset_shape": f"{X.shape[0]}x{X.shape[1]}",
        "n_relevant_features": N_RELEVANT_FEATURES,
        "n_noise_features_observed": n_noise_features,
        "n_total_features": int(X.shape[1]),
        "positive_rate": positive_rate,
        "negative_rate": float(1.0 - positive_rate),
        "expected_positive_rate": 0.5,
        "class_rate_close_to_expected": bool(
            abs(positive_rate - 0.5) <= CLASS_RATE_TOLERANCE
        ),
        "class_0_count": class_counts.get(0, 0),
        "class_1_count": class_counts.get(1, 0),
        "primitive_conditions_recomputed": bool(
            conditions.shape == (N_SAMPLES, N_RELEVANT_FEATURES)
        ),
        "label_rule_valid": bool(np.array_equal(y, direct_target)),
        "labels_binary": bool(set(values.tolist()).issubset({0, 1})),
        "n_samples_valid": bool(X.shape[0] == N_SAMPLES),
        "n_relevant_features_valid": bool(
            int(LOGICAL_TASKS[DATASET_VARIANT]["n_relevant_features"])
            == N_RELEVANT_FEATURES
        ),
        "n_noise_features_valid": bool(n_noise_features in NOISE_LEVELS),
        "n_total_features_valid": bool(
            X.shape[1] == N_RELEVANT_FEATURES + n_noise_features
        ),
        "finite_values_valid": bool(
            np.isfinite(X).all() and np.isfinite(y).all()
        ),
        "engineered_variables_absent": bool(
            X.shape[1] == N_RELEVANT_FEATURES + n_noise_features
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
            "L3 noise-sweep validation failed for "
            f"noise={n_noise_features}: {validation}"
        )
    return validation


def _create_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    successful = raw_results.loc[raw_results["status"] == "success"].copy()
    grouped = successful.groupby(
        ["model", "n_noise_features"],
        sort=True,
    )
    summary = grouped[SUMMARY_METRICS].agg(["mean", "std"])
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
        on=["model", "n_noise_features"],
        how="left",
    )


def _create_plot(summary: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name in sorted(summary["model"].unique()):
        rows = summary.loc[summary["model"] == model_name].sort_values(
            "n_noise_features"
        )
        axis.errorbar(
            rows["n_noise_features"],
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
    axis.set_xticks(NOISE_LEVELS)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Number of irrelevant features")
    axis.set_ylabel("Mean balanced accuracy")
    axis.set_title(
        "Robustness of multi-condition XOR reasoning to irrelevant features"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=300)
    plt.close(figure)


def _print_reports(
    raw_results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    table = summary.pivot(
        index="model",
        columns="n_noise_features",
        values="balanced_accuracy_mean",
    ).reindex(columns=NOISE_LEVELS)
    print("\nMean balanced accuracy:")
    print(table.round(4).to_string())

    drops = table[0] - table[8]
    print("\nBalanced-accuracy drop from noise 0 to noise 8:")
    for model_name, value in drops.sort_index().items():
        print(f"{model_name}: {value:.4f}")

    print("\nFirst noise level with mean balanced accuracy <= 0.55:")
    for model_name, row in table.sort_index().iterrows():
        qualifying = [
            noise for noise in NOISE_LEVELS if row[noise] <= 0.55
        ]
        first = str(qualifying[0]) if qualifying else "not reached"
        print(f"{model_name}: {first}")

    print("\nModel ranking at each noise level:")
    for noise in NOISE_LEVELS:
        ranking = table[noise].sort_values(ascending=False)
        text = ", ".join(
            f"{rank}. {model} ({score:.4f})"
            for rank, (model, score) in enumerate(ranking.items(), start=1)
        )
        print(f"Noise {noise}: {text}")

    successful = int((raw_results["status"] == "success").sum())
    failed = int((raw_results["status"] == "failed").sum())
    print(f"\nTotal successful runs: {successful}")
    print(f"Total failed runs: {failed}")
    print("\nExact output paths:")
    for path in (
        RAW_RESULTS_PATH,
        SUMMARY_PATH,
        VALIDATION_PATH,
        PLOT_PATH,
    ):
        print(path.resolve())


def run_logical_l3_noise_sweep(
    model_names: list[str],
    seeds: list[int],
) -> dict[str, Path]:
    """Run only L3 over the fixed irrelevant-feature sweep."""
    if tuple(seeds) != REQUIRED_SEEDS:
        raise ValueError("L3 noise sweep requires seeds 0, 1, 2, 3, and 4.")
    if set(model_names) != REQUIRED_MODELS or len(model_names) != len(
        REQUIRED_MODELS
    ):
        raise ValueError(
            "L3 noise sweep requires exactly catboost, realmlp, tabicl_v2, "
            "limix, tabpfn_v2, and tabpfn_v3."
        )

    splits: dict[tuple[int, int], dict[str, Any]] = {}
    for n_noise_features in NOISE_LEVELS:
        for seed in seeds:
            X, y = generate_logical_dataset(
                dataset_variant=DATASET_VARIANT,
                n_noise_features=n_noise_features,
                seed=seed,
                n_samples=N_SAMPLES,
            )
            indices = np.arange(N_SAMPLES)
            train_indices, test_indices = train_test_split(
                indices,
                test_size=0.25,
                random_state=seed,
                stratify=y,
            )
            validation = validate_l3_dataset_and_split(
                X,
                y,
                n_noise_features,
                train_indices,
                test_indices,
            )
            splits[(n_noise_features, seed)] = {
                "X_train": X[train_indices],
                "X_test": X[test_indices],
                "y_train": y[train_indices],
                "y_test": y[test_indices],
                "train_indices": train_indices,
                "test_indices": test_indices,
                "validation": validation,
            }
            print(
                "Validation | variant=L3 | "
                f"noise={n_noise_features} | seed={seed} | "
                f"shape={validation['dataset_shape']} | "
                f"positive_rate={validation['positive_rate']:.3f} | "
                f"label_rule_valid={validation['label_rule_valid']} | "
                f"finite={validation['finite_values_valid']} | "
                f"engineered_absent="
                f"{validation['engineered_variables_absent']}"
            )

    model_factories = get_model(model_names)
    if RAW_RESULTS_PATH.exists():
        existing = pd.read_csv(RAW_RESULTS_PATH)
        raw_rows = existing.loc[existing["status"] == "success"].to_dict(
            "records"
        )
    else:
        raw_rows = []
    validation_rows = (
        pd.read_csv(VALIDATION_PATH).to_dict("records")
        if VALIDATION_PATH.exists()
        else []
    )
    completed = {
        (str(row["model"]), int(row["n_noise_features"]), int(row["seed"]))
        for row in raw_rows
    }
    seen_model_references: list[weakref.ReferenceType[Any]] = []

    for n_noise_features in NOISE_LEVELS:
        for seed in seeds:
            split = splits[(n_noise_features, seed)]
            expected_train_hash = _indices_hash(split["train_indices"])
            expected_test_hash = _indices_hash(split["test_indices"])
            for model_name, model_factory in model_factories.items():
                run_key = (model_name, n_noise_features, seed)
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

                print(
                    f"Running {model_name} | L3 | "
                    f"noise={n_noise_features} | seed={seed}"
                )
                base_row = {
                    "dataset_module": "logical_dataset",
                    "dataset_variant": DATASET_VARIANT,
                    "logical_operator": LOGICAL_OPERATOR,
                    "n_noise_features": n_noise_features,
                    "model": model_name,
                    "seed": seed,
                    "n_samples": N_SAMPLES,
                    "n_relevant_features": N_RELEVANT_FEATURES,
                    "n_features": N_RELEVANT_FEATURES + n_noise_features,
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
                    seen_model_references.append(weakref.ref(underlying_model))
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
                    predictions_newly_generated = tracked_model.predict_calls >= 1
                    if not predictions_newly_generated:
                        raise AssertionError("No new prediction call was observed.")
                    row = {
                        **base_row,
                        **metrics,
                        "status": "success",
                        "error": "",
                    }
                    print(
                        f"Balanced accuracy={metrics['balanced_accuracy']:.4f} | "
                        f"Accuracy={metrics['accuracy']:.4f}"
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
                    "dataset_variant": DATASET_VARIANT,
                    "logical_operator": LOGICAL_OPERATOR,
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
                    all(dataset_validation[flag] for flag in required_flags)
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
    _create_plot(summary)
    _print_reports(raw_results, summary)
    return {
        "raw_results": RAW_RESULTS_PATH,
        "summary": SUMMARY_PATH,
        "validation": VALIDATION_PATH,
        "plot": PLOT_PATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-seed L3 XOR irrelevant-feature sweep."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(REQUIRED_MODELS),
        help="The exact six-model comparison used in the reported experiment.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(REQUIRED_SEEDS),
        help="The reported experiment uses seeds 0, 1, 2, 3, and 4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_logical_l3_noise_sweep(arguments.models, arguments.seeds)
