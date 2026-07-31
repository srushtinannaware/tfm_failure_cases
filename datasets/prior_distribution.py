"""
Prior Distribution Mismatch Dataset:
Tests model robustness against feature distributions that violate the 
synthetic Gaussian/smooth priors used to train Tabular Foundation Models.

Uses Monotonic Probability Integral Transforms:
The underlying signal rank relationship is IDENTICAL across all variants.
Because CatBoost is rank-invariant, it maintains ~90%+ accuracy everywhere.
TFMs degrade on Pareto/Bimodal/Poisson because raw feature scales and step 
discontinuities mess up continuous feature embeddings and LayerNorms.

Usage 1 (Standard main.py harness):
    python main.py --dataset prior_distribution --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

Usage 2 (Standalone module check matching deep_causal_chain.py pattern):
    python -m datasets.prior_distribution
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np
from scipy.special import erf
from scipy.stats import poisson

N_TRAIN = 1000
N_TEST = 200
N_FEATURES = 6
MASTER_SEED_OFFSET = 300


def _apply_label_rule(S: np.ndarray) -> np.ndarray:
    """Label is determined purely by the standardized underlying signal."""
    score = S[:, 0] + S[:, 1] + S[:, 2]
    return (score > np.median(score)).astype(int)


# ---------------------------------------------------------------------------
# main.py entry point
# ---------------------------------------------------------------------------
def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. Generate Latent Gaussian Signal Matrix S
    S = rng.normal(0, 1, size=(n_total, N_FEATURES))
    y = _apply_label_rule(S)

    # Variant 1: Control (Standard Gaussian)
    datasets["prior_gaussian_control"] = (S.copy(), y)

    # Variant 2: Heavy-Tailed Pareto Distribution (Monotonic transform of S)
    u = 0.5 * (1.0 + erf(S / np.sqrt(2.0)))
    u = np.clip(u, 1e-5, 1.0 - 1e-5)
    pareto_mag = (1.0 - np.abs(2.0 * u - 1.0)) ** (-1.0 / 0.7) - 1.0
    X_pareto = np.sign(S) * pareto_mag
    datasets["prior_pareto_heavy_tail"] = (X_pareto, y)

    # Variant 3: Bimodal Cluster Distribution
    X_bimodal = np.sign(S) * (np.abs(S) + 4.0)
    datasets["prior_bimodal_clusters"] = (X_bimodal, y)

    # Variant 4: Discrete Poisson Count Features
    X_poisson = poisson.ppf(u, mu=3.0).astype(float)
    datasets["prior_poisson_counts"] = (X_poisson, y)

    return datasets


# ---------------------------------------------------------------------------
# Dedicated runner matching deep_causal_chain.py pattern
# ---------------------------------------------------------------------------
def run_prior_distribution_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
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
    csv_path = results_dir / "prior_distribution_check_results.csv"

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
    print("Running Prior Distribution Benchmark")
    print("=" * 70)
    run_prior_distribution_check(model_names=models_to_run)