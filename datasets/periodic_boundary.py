"""
High-Frequency Periodic Decision Boundary Dataset:
Exploits spectral bias in in-context foundation models. TFMs favor smooth,
low-frequency decision surfaces. High-frequency oscillations violate their
prior, while CatBoost naturally approximates them using axis-aligned splits.

Variants:
  - periodic_low_frequency:  sin(1 * x0) > 0 (Easy baseline)
  - periodic_mid_frequency:  sin(4 * x0) > 0 (Medium difficulty)
  - periodic_high_frequency: sin(10 * x0) > 0 (Exposes TFM prior smoothing)
  - periodic_2d_checkerboard: sin(6 * x0) * cos(6 * x1) > 0 (Complex 2D grid)

Usage:
    python main.py --dataset periodic_boundary --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
"""

from __future__ import annotations

import numpy as np

N_SAMPLES = 1500
N_FEATURES = 5
MASTER_SEED_OFFSET = 400


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Sample uniformly across continuous spatial domain
    X = rng.uniform(-np.pi * 2, np.pi * 2, size=(N_SAMPLES, N_FEATURES))

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


if __name__ == "__main__":
    # Standalone sanity check: verify spatial range and grid balance
    datasets = get_datasets(seed=0)
    print("=== Periodic Boundary Sanity Check ===")
    for variant, (X, y) in datasets.items():
        print(f"\nVariant: {variant}")
        print(f"  X shape: {X.shape} | y shape: {y.shape}")
        print(f"  X domain range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  Label balance (% positive): {y.mean() * 100:.1f}%")