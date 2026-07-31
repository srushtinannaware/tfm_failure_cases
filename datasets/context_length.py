"""
Context Length & Attention Dilution Dataset:
Tests transformer in-context attention bottlenecks against standard training scaling.

Variants:
  - row_poisoning_0pct:  Clean dataset with 1,000 signal rows.
  - row_poisoning_50pct: 1,000 signal rows + 1,000 uninformative noise rows.
  - row_poisoning_80pct: 1,000 signal rows + 4,000 uninformative noise rows.

Usage:
    python main.py --dataset context_length --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
"""

from __future__ import annotations

import numpy as np

N_SIGNAL_SAMPLES = 1000
N_FEATURES = 8
MASTER_SEED_OFFSET = 500


def _generate_clean_signal(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0, 1, size=(n, N_FEATURES))
    # Non-linear XOR-style rule across first two features
    score = X[:, 0] * X[:, 1] + X[:, 2]
    y = (score > 0).astype(int)
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Base clean dataset
    X_clean, y_clean = _generate_clean_signal(N_SIGNAL_SAMPLES, rng)
    datasets["attention_clean_base"] = (X_clean, y_clean)

    # Variant 1: 50% Poisoned Rows (1000 clean + 1000 pure random noise rows)
    X_noise_1000 = rng.normal(0, 1, size=(1000, N_FEATURES))
    y_noise_1000 = rng.choice([0, 1], size=1000)

    X_50 = np.vstack([X_clean, X_noise_1000])
    y_50 = np.concatenate([y_clean, y_noise_1000])
    
    # Shuffle to interleave noise and signal rows
    perm_50 = rng.permutation(len(y_50))
    datasets["attention_dilution_50pct"] = (X_50[perm_50], y_50[perm_50])

    # Variant 2: 80% Poisoned Rows (1000 clean + 4000 pure random noise rows)
    X_noise_4000 = rng.normal(0, 1, size=(4000, N_FEATURES))
    y_noise_4000 = rng.choice([0, 1], size=4000)

    X_80 = np.vstack([X_clean, X_noise_4000])
    y_80 = np.concatenate([y_clean, y_noise_4000])
    
    perm_80 = rng.permutation(len(y_80))
    datasets["attention_dilution_80pct"] = (X_80[perm_80], y_80[perm_80])

    return datasets


if __name__ == "__main__":
    # Standalone sanity check: confirm sample growth across poisoned variants
    datasets = get_datasets(seed=0)
    print("=== Context Length / Row Poisoning Sanity Check ===")
    for variant, (X, y) in datasets.items():
        print(f"\nVariant: {variant}")
        print(f"  X shape: {X.shape} | y shape: {y.shape}")
        print(f"  Label balance (% positive): {y.mean() * 100:.1f}%")