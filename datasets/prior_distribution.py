"""
Prior Distribution Mismatch Dataset:
Tests model robustness against feature distributions that violate the 
synthetic Gaussian/smooth priors used to train Tabular Foundation Models.

Variants:
  - control_gaussian: Standard normal features (within prior).
  - pareto_heavy_tail: Extremely heavy-tailed Pareto distribution.
  - bimodal_clusters: Bimodal Gaussian mixture distribution.
  - poisson_counts: Non-continuous discrete integer count features.

The underlying decision logic remains fixed across all variants:
  label = 1 if (X0 + X1 + X2 > threshold) else 0

Usage:
    python main.py --dataset prior_distribution --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
"""

from __future__ import annotations

import numpy as np

N_SAMPLES = 1200
N_FEATURES = 6
MASTER_SEED_OFFSET = 300


def _apply_label_rule(X_standardized: np.ndarray) -> np.ndarray:
    """Label is determined purely by the standardized underlying signal."""
    score = X_standardized[:, 0] + X_standardized[:, 1] + X_standardized[:, 2]
    return (score > np.median(score)).astype(int)


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. Control (Standard Gaussian)
    X_base = rng.normal(0, 1, size=(N_SAMPLES, N_FEATURES))
    y = _apply_label_rule(X_base)
    datasets["prior_gaussian_control"] = (X_base, y)

    # 2. Heavy-Tailed Pareto Distribution (Extreme tail values)
    # Underlying signal preserved via uniform quantile mapping
    pareto_raw = rng.pareto(a=0.7, size=(N_SAMPLES, N_FEATURES))
    # Apply sign flip randomly to create two-sided heavy tails
    signs = rng.choice([-1, 1], size=(N_SAMPLES, N_FEATURES))
    X_pareto = pareto_raw * signs
    datasets["prior_pareto_heavy_tail"] = (X_pareto, y)

    # 3. Bimodal Clusters
    cluster_ids = rng.choice([0, 1], size=(N_SAMPLES, N_FEATURES))
    X_bimodal = rng.normal(loc=np.where(cluster_ids == 1, 4.0, -4.0), scale=0.5)
    datasets["prior_bimodal_clusters"] = (X_bimodal, y)

    # 4. Poisson Count Features
    X_poisson = rng.poisson(lam=3.0, size=(N_SAMPLES, N_FEATURES)).astype(float)
    datasets["prior_poisson_counts"] = (X_poisson, y)

    return datasets

if __name__ == "__main__":
    # Standalone sanity check: print feature ranges and label balances
    datasets = get_datasets(seed=0)
    print("=== Prior Distribution Sanity Check ===")
    for variant, (X, y) in datasets.items():
        print(f"\nVariant: {variant}")
        print(f"  X shape: {X.shape} | y shape: {y.shape}")
        print(f"  X min: {X.min():.2f} | X max: {X.max():.2f}")
        print(f"  Label balance (% positive): {y.mean() * 100:.1f}%")