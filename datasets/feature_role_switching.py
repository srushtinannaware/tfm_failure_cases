"""
Feature Grouping Distance Dataset (formerly "role switching") — rebuilt
twice now. First rebuild dropped noise and decoupled role-branching from
distance, but a sanity check with a real gradient booster (sklearn
GradientBoostingClassifier standing in for CatBoost, since trees are
provably column-order-blind) showed accuracy swinging randomly by seed
(0.48 to 0.92) regardless of distance at pool_size=60/N_TRAIN=600 — a
needle-in-haystack search problem (which 2 of 60 columns interact, out of
~1770 candidate pairs), not a real distance effect. That noise floor
would have swamped anything TabPFN-v3-specific.

Fixed by shrinking the candidate pool and raising sample size until the
search problem is reliably solved regardless of column distance — checked
empirically (not just via an oracle-ceiling calculation, which was
insufficient last time):

    pool_size=20, N_TRAIN=1200, noise_std=0.05:
        GradientBoostingClassifier gets 90-95% across 5 seeds, at d=1
        AND d=19 (opposite ends of the pool) — i.e. distance genuinely
        doesn't matter for a tree, as it shouldn't, confirming this is
        now a clean baseline to test the (potentially different)
        triplet-grouping story on TabPFN-3/TabICL-v2 against.

    role_sanity K=2 variants needed 1200 samples/role (2400 total) to
    reach ~87-93% reliably; 300-600/role (what the earlier version used)
    was NOT reliable (0.57-0.93 spread across seeds) — that alone likely
    explains a good chunk of the flat-chance results by role_switch_5roles
    in your last graph.

TODO(kate): run role_sanity_2roles_adjacent FIRST. If CatBoost/RealMLP/
TabPFN-v2 aren't comfortably >85% there, something in your actual
models.py config (not this dataset) is still limiting search — these
numbers were validated with a generic sklearn GBM, not your exact
CatBoost hyperparameters, so there could still be a gap between the two.

TODO(kate): distance_sweep_* has NO role branching at all (single fixed
pair) — that's deliberate, to isolate the grouping-distance question
from the branching-search question that broke the last two versions.
Reintroduce roles only after distance_sweep_* shows a clean result on
its own.

TODO(kate): TabPFN-3 and TabICL-v2 reportedly share the triplet-grouping
mechanism per their technical reports. If TabICL-v2 degrades alongside
TabPFN-v3 on distance_sweep_* while CatBoost/RealMLP/TabPFN-v2 don't,
that's still the result you're after (a shared-architecture robustness
gap) — not a failed experiment.

Variants:
  - role_sanity_2roles_adjacent / role_sanity_2roles_farsplit:
        run first. K=2, 1200 samples/role, noise=0.05. Identical except
        column placement (adjacent vs. opposite ends of a 20-col pool).
  - distance_sweep_d1 / d3 / d6 / d10 / d14 / d19:
        THE experiment. Single fixed pair, pool_size=20 held constant,
        column a fixed at index 0, column b at index d. Only distance
        between the interacting columns changes.

Usage:
    python main.py --dataset feature_role_switching --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
    python -m datasets.feature_role_switching
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

MASTER_SEED_OFFSET = 600
INTERACTION_NOISE_STD = 0.05  # oracle ceiling ~94.8%, checked numerically

# distance-sweep settings (primary experiment) — validated with a real
# gradient booster to reliably solve the search problem at every distance
DIST_POOL_SIZE = 20
DIST_N_TRAIN = 1200
DIST_N_TEST = 300
DIST_VALUES = (1, 3, 6, 10, 14)

# role-sanity settings (run first)
SANITY_TRAIN_PER_ROLE = 1200
SANITY_TEST_PER_ROLE = 200
SANITY_POOL_SIZE = 20


def _make_pair_interaction(n: int, pool_size: int, col_a: int, col_b: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0, 1, size=(n, pool_size))
    a = X[:, col_a]
    b = X[:, col_b]
    score = a * b + rng.normal(0, INTERACTION_NOISE_STD, n)
    y = (score > 0).astype(int)
    return X, y


def _make_role_switch(n: int, n_roles: int, pool_size: int, rng, far_split: bool = False) -> tuple[np.ndarray, np.ndarray]:
    X_pool = rng.normal(0, 1, size=(n, pool_size))
    role = rng.integers(0, n_roles, size=n)

    if far_split:
        col_a = role
        col_b = pool_size - 1 - role
    else:
        col_a = role * 2
        col_b = role * 2 + 1

    a = X_pool[np.arange(n), col_a]
    b = X_pool[np.arange(n), col_b]
    score = a * b + rng.normal(0, INTERACTION_NOISE_STD, n)
    y = (score > 0).astype(int)

    X = np.hstack([X_pool, role.reshape(-1, 1).astype(float)])
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns variant_name -> (X, y) where X and y are already
    concatenated train+test so main.py can do its own split."""
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # --- sanity checks: run these first ---
    n_roles = 2
    n_train = SANITY_TRAIN_PER_ROLE * n_roles
    n_test = SANITY_TEST_PER_ROLE * n_roles
    X, y = _make_role_switch(n_train + n_test, n_roles, SANITY_POOL_SIZE, rng, far_split=False)
    datasets["role_sanity_2roles_adjacent"] = (X, y)

    X, y = _make_role_switch(n_train + n_test, n_roles, SANITY_POOL_SIZE, rng, far_split=True)
    datasets["role_sanity_2roles_farsplit"] = (X, y)

    # --- primary experiment: pure distance sweep, no branching ---
    for d in DIST_VALUES:
        X, y = _make_pair_interaction(DIST_N_TRAIN + DIST_N_TEST, DIST_POOL_SIZE, col_a=0, col_b=d, rng=rng)
        datasets[f"distance_sweep_d{d}"] = (X, y)

    return datasets


def run_role_switch_check(model_names: list[str], seeds: list[int] = (0, 1, 2)) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
            n_train = int(len(X) * 0.8)
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]

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
    csv_path = results_dir / "feature_role_switching_results.csv"

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
    print("Running Feature Grouping Distance Benchmark")
    print("=" * 70)
    run_role_switch_check(model_names=models_to_run)