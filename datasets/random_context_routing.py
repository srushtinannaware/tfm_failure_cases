"""
Random Context Routing Dataset:
Tests EXACT few-shot retrieval from context under increasing key
cardinality, instead of smooth function approximation.

Each row carries a high-cardinality categorical "key" (0..M-1) and a
block of numeric decoy features carrying NO signal. The true label for
each key is drawn from a fresh random lookup table PER SEED — there is
no smooth relationship between nearby key values, only a lookup that
must be read off from context. Keys are shared between train and test
within a run (a test key either appeared in train or it didn't — there
is no way to interpolate for unseen keys, which is intentional).

As M grows with N_TRAIN fixed, shots-per-key shrinks toward 1 and then
below 1 (unseen-key regime). This targets ICL attention's soft nearest-
neighbor retrieval specifically: with many near-duplicate context
tokens and few repeats per exact key, retrieval competes against noise
the same way it does over feature columns in the existing attention-
dilution dataset, except here the dilution is over CONTEXT ROWS.
CatBoost's native categorical handling (effectively per-category
target statistics) is expected to degrade more gracefully at moderate
M and only fail once shots-per-key genuinely hits ~1, same as anyone.

TODO(kate): if your models.py passes `key` through as a raw float/int
column rather than declaring it categorical, CatBoost won't get its
native categorical treatment and this stops being a fair test of the
"trees degrade gracefully" half of the hypothesis — check how
categorical column indices get passed into evaluate_classifier.

Variants:
  - routing_M10 / M50 / M200 / M800 / M2000: number of distinct keys,
    N_TRAIN fixed at 1000, so mean shots-per-key drops from ~100 to
    <1 (M2000 deliberately pushes past the point where the task is
    solvable at all, as a sanity floor for the sweep).

Usage:
    python main.py --dataset random_context_routing --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
    python -m datasets.random_context_routing
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

N_TRAIN = 1000
N_TEST = 200
N_DECOY_FEATURES = 10
MASTER_SEED_OFFSET = 800
LABEL_FLIP_PROB = 0.02


def _make_routing(n: int, n_keys: int, n_decoy: int, rng) -> tuple[np.ndarray, np.ndarray]:
    keys = rng.integers(0, n_keys, size=n)
    lookup = rng.integers(0, 2, size=n_keys)
    y = lookup[keys]

    flips = rng.random(n) < LABEL_FLIP_PROB
    y = np.where(flips, 1 - y, y)

    decoys = rng.normal(0, 1, size=(n, n_decoy))
    X = np.hstack([keys.reshape(-1, 1).astype(float), decoys])
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets = {}

    for n_keys in (10, 50, 200, 800, 2000):
        X, y = _make_routing(n_total, n_keys, N_DECOY_FEATURES, rng)
        datasets[f"routing_M{n_keys}"] = (X, y)

    return datasets


def run_context_routing_check(model_names: list[str], seeds: list[int] = (0, 1, 2)) -> list[dict]:
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
    csv_path = results_dir / "random_context_routing_results.csv"

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
    print("Running Random Context Routing Benchmark")
    print("=" * 70)
    run_context_routing_check(model_names=models_to_run)
