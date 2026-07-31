"""
Context Length & Attention Dilution Dataset:
Tests transformer in-context attention bottlenecks against standard training scaling.

Variants:
  - attention_clean_base:   1,000 clean signal training rows.
  - attention_dilution_50:  1,000 clean + 1,000 noisy training rows (50% signal).
  - attention_dilution_80:  1,000 clean + 4,000 noisy training rows (20% signal).

Test set stays strictly constant across all variants at N_TEST = 200 clean rows.

Usage 1 (Standard main.py harness):
    python main.py --dataset context_length --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

Usage 2 (Standalone module check matching deep_causal_chain.py pattern):
    python datasets/context_length.py
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path

import numpy as np

N_TRAIN = 1000
N_TEST = 200
N_FEATURES = 8
MASTER_SEED_OFFSET = 500


def _generate_clean_signal(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0, 1, size=(n, N_FEATURES))
    score = X[:, 0] * X[:, 1] + X[:, 2]
    y = (score > 0).astype(int)
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns full (X, y) combined arrays per variant.
    The first N_TRAIN (or scaled N_TRAIN) rows form training data,
    while the final N_TEST (200) rows form the constant clean test set.
    """
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Base clean train (1000) and clean test (200)
    X_clean_train, y_clean_train = _generate_clean_signal(N_TRAIN, rng)
    X_clean_test, y_clean_test = _generate_clean_signal(N_TEST, rng)

    # 1. Clean Base Variant
    X_base = np.vstack([X_clean_train, X_clean_test])
    y_base = np.concatenate([y_clean_train, y_clean_test])
    datasets["attention_clean_base"] = (X_base, y_base)

    # 2. 50% Dilution (1000 clean train + 1000 noisy train + 200 clean test)
    X_noise_1000 = rng.normal(0, 1, size=(1000, N_FEATURES))
    y_noise_1000 = rng.choice([0, 1], size=1000)

    X_train_50 = np.vstack([X_clean_train, X_noise_1000])
    y_train_50 = np.concatenate([y_clean_train, y_noise_1000])
    perm_50 = rng.permutation(len(y_train_50))
    X_train_50, y_train_50 = X_train_50[perm_50], y_train_50[perm_50]

    X_50 = np.vstack([X_train_50, X_clean_test])
    y_50 = np.concatenate([y_train_50, y_clean_test])
    datasets["attention_dilution_50pct"] = (X_50, y_50)

    # 3. 80% Dilution (1000 clean train + 4000 noisy train + 200 clean test)
    X_noise_4000 = rng.normal(0, 1, size=(4000, N_FEATURES))
    y_noise_4000 = rng.choice([0, 1], size=4000)

    X_train_80 = np.vstack([X_clean_train, X_noise_4000])
    y_train_80 = np.concatenate([y_clean_train, y_noise_4000])
    perm_80 = rng.permutation(len(y_train_80))
    X_train_80, y_train_80 = X_train_80[perm_80], y_train_80[perm_80]

    X_80 = np.vstack([X_train_80, X_clean_test])
    y_80 = np.concatenate([y_train_80, y_clean_test])
    datasets["attention_dilution_80pct"] = (X_80, y_80)

    return datasets


def run_context_length_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
) -> list[dict]:
    """
    Evaluates requested models across attention dilution variants,
    keeping the test set fixed at N_TEST = 200 clean samples.
    """
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
            # Test set is ALWAYS the last N_TEST (200) rows across all variants
            X_train, y_train = X[:-N_TEST], y[:-N_TEST]
            X_test, y_test = X[-N_TEST:], y[-N_TEST:]

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
    csv_path = results_dir / "context_length_check_results.csv"

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
    print("Running Context Length & Attention Dilution Benchmark")
    print("=" * 70)
    run_context_length_check(model_names=models_to_run)