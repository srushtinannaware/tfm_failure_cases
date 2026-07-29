"""
Extreme feature-scale mismatch: tests whether models correctly infer that
raw feature MAGNITUDE does not imply relative IMPORTANCE, when features are
left unnormalized at wildly different scales (~1e-3 vs ~1e2 vs ~1e5).

Design: the TRUE underlying signal is identical in both variants below -
same standardized combination (a + b + c), same label rule, same noise.
Only the raw scale each feature is presented in differs. This isolates
scale distortion as the variable being tested, not task difficulty.

Hypothesis: TabPFN/TabICL are pretrained on synthetic priors that assume
roughly "typical" feature ranges; values far outside that prior's training
distribution may degrade in-context performance. CatBoost is scale-invariant
by construction (splits on order statistics, not raw magnitude), and
RealMLP-TD's "TD" preprocessing pipeline may normalize internally - both
should be relatively robust. This directly complements the S1/S2
shortcut-learning angle: it tests a different kind of prior mismatch
(distributional, not causal/spurious-correlation).

Usage:
    python main.py --dataset feature_scale_mismatch --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
"""

from __future__ import annotations

import numpy as np

N_TRAIN = 1000
N_TEST = 200
MASTER_SEED_OFFSET = 200  # separate stream from other dataset modules

LABEL_NOISE = 0.02

# Scale multipliers applied to the standardized (a, b, c) signal before
# exposing to the model. "control" keeps scales comparable; "extreme"
# spreads them across 8 orders of magnitude.
CONTROL_SCALES = {"a": 1.0, "b": 1.0, "c": 1.0}
EXTREME_SCALES = {"a": 1e-3, "b": 1e2, "c": 1e5}


def _generate(n_total: int, scales: dict[str, float], rng) -> tuple[np.ndarray, np.ndarray]:
    a = rng.normal(0, 1, n_total)
    b = rng.normal(0, 1, n_total)
    c = rng.normal(0, 1, n_total)

    # true signal computed on the STANDARDIZED values, equal weights -
    # scale is applied only afterward, to what the model actually sees
    score = a + b + c
    T = np.median(score)
    y = (score > T).astype(int)

    # label noise, same convention as generate_datasets.py
    n_flip = int(round(LABEL_NOISE * n_total))
    flip_idx = rng.choice(n_total, size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    X = np.stack([a * scales["a"], b * scales["b"], c * scales["c"]], axis=1)
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    X_control, y_control = _generate(n_total, CONTROL_SCALES, rng)
    datasets["scale_mismatch_control"] = (X_control, y_control)

    X_extreme, y_extreme = _generate(n_total, EXTREME_SCALES, rng)
    datasets["scale_mismatch_extreme"] = (X_extreme, y_extreme)

    return datasets


if __name__ == "__main__":
    # Quick standalone sanity check: confirm control and extreme variants
    # carry the IDENTICAL underlying signal (same label array, since scale
    # is applied after label assignment) - only raw feature magnitude
    # differs. This is what makes the comparison fair.
    rng = np.random.default_rng(0 + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    X_c, y_c = _generate(n_total, CONTROL_SCALES, rng)

    rng2 = np.random.default_rng(0 + MASTER_SEED_OFFSET)
    X_e, y_e = _generate(n_total, EXTREME_SCALES, rng2)

    print("Labels identical between control/extreme (same seed):", np.array_equal(y_c, y_e))
    print("Control feature ranges:", X_c.min(axis=0), "to", X_c.max(axis=0))
    print("Extreme feature ranges:", X_e.min(axis=0), "to", X_e.max(axis=0))
    print("Label balance:", y_c.mean())