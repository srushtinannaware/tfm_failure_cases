"""
High-Frequency Periodic Decision Boundary Dataset:
Exploits spectral bias in in-context foundation models. TFMs favor smooth,
low-frequency decision surfaces. High-frequency oscillations violate their
prior, while CatBoost naturally approximates them using axis-aligned splits.

Variants:
  - periodic_low_freq:        sin(1 * x0) > 0 (Easy baseline)
  - periodic_mid_freq:        sin(4 * x0) > 0 (Medium difficulty)
  - periodic_high_freq:       sin(10 * x0) > 0 (Exposes TFM prior smoothing)
  - periodic_2d_checkerboard: sin(6 * x0) * cos(6 * x1) > 0 (Complex 2D grid)

Usage 1 (Standard main.py harness):
    python main.py --dataset periodic_boundary --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

Usage 2 (Standalone module check matching deep_causal_chain.py pattern):
    python datasets/periodic_boundary.py
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path

import numpy as np

N_TRAIN = 1000
N_TEST = 200
N_FEATURES = 5
MASTER_SEED_OFFSET = 400


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    X = rng.uniform(-np.pi * 2, np.pi * 2, size=(n_total, N_FEATURES))

    # Variant 1: Low Frequency (Smooth)
    y_low = (np.sin(1.0 * X[:, 0]) > 0).astype(int)
    datasets["periodic_low_freq"] = (X, y_low)

    # Variant 2: Mid Frequency
    y_mid = (np.sin(4.0 * X[:, 0]) > 0).astype(int)
    datasets["periodic_mid_freq"] = (X, y_mid)

    # Variant 3: High Frequency (Violates TFM smoothness prior)
    y_high = (np.sin(10.0 * X[:, 0]) > 0).astype(int)
    datasets["periodic_high_freq"] = (X, y_high)

    # Variant 4: 2D Oscillating Interaction (Checkerboard pattern)
    y_2d = (np.sin(6.0 * X[:, 0]) * np.cos(6.0 * X[:, 1]) > 0).astype(int)
    datasets["periodic_2d_checkerboard"] = (X, y_2d)

    return datasets


def run_periodic_boundary_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
) -> list[dict]:
    """
    Evaluates requested models across all periodic boundary variants,
    prints metrics live, and saves results to results/periodic_boundary_check_results.csv.
    """
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
            # Explicit split matching N_TRAIN (1000) and N_TEST (200)
            X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
            X_test, y_test = X[N_TRAIN:], y[N_TRAIN:]

            for model_name, factory in model_factories.items():
                print(f"[{variant_name}] {model_name} (seed={seed})...")

                base_row = {
                    "dataset_variant": variant_name,
                    "model": model_name,
                    "seed": seed,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "train_samples": int(X_train.shape[0]),
                    "test_samples": int(X_test.shape[0]),
                }

                try:
                    metrics = evaluate_classifier(
                        model=factory(),
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                    )

                    row = {**base_row, **metrics, "status": "success", "error": ""}
                    print(
                        f"  Accuracy={metrics['accuracy']:.4f} | "
                        f"F1={metrics['f1_macro']:.4f} | "
                        f"Log loss={metrics['log_loss']:.4f} | "
                        f"Time={metrics['total_seconds']:.2f}s"
                    )

                except Exception as error:
                    row = {**base_row, "status": "failed", "error": str(error)}
                    print(f"  Failed: {error}")

                rows.append(row)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "periodic_boundary_check_results.csv"

    all_columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in all_columns:
                all_columns.append(column)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {csv_path}")
    return rows


if __name__ == "__main__":
    models_to_run = ["catboost", "realmlp", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]

    print("=" * 70)
    print("Running Periodic Boundary Benchmark")
    print("=" * 70)
    run_periodic_boundary_check(model_names=models_to_run)