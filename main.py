"""Common classification benchmark pipeline.

This module provides functions to run classification benchmarks on datasets defined in the failure_cases/ directory. It loads dataset, runs selected models, calculates metrics, and saves the results to CSV files.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from metrics import evaluate_classifier
from models import available_models, get_model


RESULTS_DIR = Path("results")


def load_dataset_module(
    dataset_name: str,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Dynamically load failure_cases/<dataset_name>.py.

    The dataset file must contain:
        get_datasets(seed) -> {"variant_name": (X, y)}
    """
    module = importlib.import_module(f"failure_cases.{dataset_name}")

    if not hasattr(module, "get_datasets"):
        raise AttributeError(
            f"failure_cases/{dataset_name}.py must define get_datasets(seed)."
        )

    datasets = module.get_datasets(seed)

    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(
            "get_datasets() must return a non-empty dictionary."
        )

    validated: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for variant_name, dataset in datasets.items():
        if not isinstance(variant_name, str) or not variant_name.strip():
            raise ValueError("Dataset variant names must be non-empty strings.")
        if not isinstance(dataset, (tuple, list)) or len(dataset) != 2:
            raise ValueError(
                f"{variant_name}: expected an (X, y) pair."
            )

        X, y = map(np.asarray, dataset)
        y = y.reshape(-1)

        if X.ndim != 2:
            raise ValueError(f"{variant_name}: X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError(
                f"{variant_name}: X and y have different lengths."
            )
        if len(X) == 0:
            raise ValueError(f"{variant_name}: dataset cannot be empty.")
        if np.unique(y).size < 2:
            raise ValueError(
                f"{variant_name}: classification requires at least two classes."
            )

        validated[variant_name] = (X, y)

    return validated


def save_results(
    rows: list[dict[str, Any]],
    dataset_module_name: str,
) -> Path:
    """Save all model results to one CSV, merging with any existing rows
    on disk instead of overwriting them. Re-running with a different
    --models subset (or after a crash partway through) no longer erases
    previously completed results for the same dataset module."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{dataset_module_name}_results.csv"

    # Load whatever already exists for this dataset module
    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))

    # A run is uniquely identified by variant + model + seed. New rows
    # replace old ones with the same key (e.g. a fixed/re-run result
    # correctly supersedes a stale "failed" row); anything not touched
    # in this invocation is kept as-is.
    key_fields = ("dataset_variant", "model", "seed")

    def row_key(row: dict[str, Any]) -> tuple:
        return tuple(row.get(field) for field in key_fields)

    merged: dict[tuple, dict[str, Any]] = {
        row_key(row): row for row in existing_rows
    }
    for row in rows:
        merged[row_key(row)] = row

    combined_rows = list(merged.values())

    all_columns: list[str] = []
    for row in combined_rows:
        for column in row:
            if column not in all_columns:
                all_columns.append(column)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(combined_rows)

    return csv_path


def create_accuracy_plot(
    rows: list[dict[str, Any]],
    dataset_module_name: str,
) -> Path | None:
    """Plot accuracy across dataset variants."""
    successful_rows = [
        row for row in rows if row["status"] == "success"
    ]

    if not successful_rows:
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variant_names = list(
        dict.fromkeys(row["dataset_variant"] for row in successful_rows)
    )

    model_names = sorted(
        {row["model"] for row in successful_rows}
    )

    for model_name in model_names:
        model_rows = {
            row["dataset_variant"]: row
            for row in successful_rows
            if row["model"] == model_name
        }

        x_values = []
        y_values = []

        for index, variant_name in enumerate(variant_names):
            if variant_name in model_rows:
                x_values.append(index)
                y_values.append(
                    float(model_rows[variant_name]["accuracy"])
                )

        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=model_name,
        )

    plt.xticks(
        range(len(variant_names)),
        variant_names,
        rotation=35,
        ha="right",
    )
    plt.xlabel("Dataset variant")
    plt.ylabel("Test accuracy")
    plt.title(f"Model comparison: {dataset_module_name}")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_DIR / f"{dataset_module_name}_accuracy.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    return plot_path


def run_benchmark(
    dataset_name: str,
    model_names: list[str],
    seeds: list[int],
    test_size: float,
) -> list[dict[str, Any]]:
    """Run selected models on the selected dataset module."""
    model_factories = get_model(model_names)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        dataset_variants = load_dataset_module(
            dataset_name=dataset_name,
            seed=seed,
        )

        for variant_name, (X, y) in dataset_variants.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=seed,
                stratify=y,
            )

            print("\n" + "=" * 70)
            print(
                f"Dataset: {variant_name} | "
                f"Seed: {seed} | "
                f"Rows: {len(X)} | Features: {X.shape[1]}"
            )
            print("=" * 70)

            for model_name, model_factory in model_factories.items():
                print(f"Running {model_name}...")

                base_row: dict[str, Any] = {
                    "dataset_module": dataset_name,
                    "dataset_variant": variant_name,
                    "model": model_name,
                    "seed": seed,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "train_samples": int(X_train.shape[0]),
                    "test_samples": int(X_test.shape[0]),
                }

                try:
                    model = model_factory()

                    metrics = evaluate_classifier(
                        model=model,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                    )

                    row = {
                        **base_row,
                        **metrics,
                        "status": "success",
                        "error": "",
                    }

                    print(
                        f"Accuracy={metrics['accuracy']:.4f} | "
                        f"F1={metrics['f1_macro']:.4f} | "
                        f"Log loss={metrics['log_loss']:.4f} | "
                        f"Time={metrics['total_seconds']:.2f}s"
                    )

                except Exception as error:
                    row = {
                        **base_row,
                        "status": "failed",
                        "error": str(error),
                    }

                    print(f"Failed: {error}")

                rows.append(row)

                # Save progress after every run.
                save_results(rows, dataset_name)

    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected classifiers on one dataset module."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset filename without .py, for example parity_dataset."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["tabpfn_v2", "tabicl_v2"],
        help=(
            "Names of models to run. Available: "
            + ", ".join(available_models())
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Random seeds.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction reserved for testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    rows = run_benchmark(
        dataset_name=args.dataset,
        model_names=args.models,
        seeds=args.seeds,
        test_size=args.test_size,
    )

    csv_path = save_results(rows, args.dataset)
    plot_path = create_accuracy_plot(rows, args.dataset)

    print("\nBenchmark finished.")
    print(f"CSV: {csv_path}")

    if plot_path is not None:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
