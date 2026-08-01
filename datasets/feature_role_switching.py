"""
Feature Role Switching Dataset:
Tests row-conditional feature selection — whether a model can use a
"role" indicator to know WHICH pair of columns is the signal for a
given row, while every other column (including pairs that ARE signal
for other roles) is pure noise for that row.

Motivation (grounded in the TabPFN-3 / TabICL-v2 technical reports,
not assumed): TabPFN-3 and TabICL-v2 group each feature together with
its two cyclically-shifted column-index neighbors into a fixed triplet
BEFORE row aggregation (TabICL's grouping scheme, adopted by TabPFN-3).
If the two columns that interact for a given role are never co-located
in the same triplet, the interaction has to be reconstructed from
already-compressed row embeddings rather than read off directly. Trees
split on any column pair regardless of index distance, and CatBoost
can just add more splits as roles grow; an MLP sees the raw row too.
This gives an architecture-specific, checkable hypothesis rather than
a vague "attention gets diluted" story.

TODO(kate): confirm your models.py wrapper doesn't reorder/shuffle
columns before calling tabpfn_v3/tabicl_v2 — if it does, the
`farsplit` variant's premise (max index distance -> different triplet)
is silently defeated. Also confirm which TabPFN/TabICL version string
in models.py actually maps to the triplet-grouping architecture
(TabPFN-3 uses it; earlier v2.x alternates row/feature attention
instead and this whole mechanism doesn't apply to it).

Variants:
  - role_switch_2roles / 5roles / 10roles / 20roles:
        role given as an explicit categorical feature; K grows, so the
        number of conditional branches a model must represent grows.
  - role_switch_10roles_farsplit:
        same K=10, but the two signal columns for each role are placed
        at maximum index distance apart (col k and pool_size-1-k), so
        they can never land in the same triplet-neighbor group. Same
        difficulty for trees/MLP as role_switch_10roles; should be
        selectively harder for triplet-grouped models if the grouping
        hypothesis is real. Compare this against role_switch_10roles
        directly — that's the actual test, not the accuracy in
        isolation.
  - role_switch_10roles_latent:
        role is NOT given as an explicit feature; it's coded in 4 noisy
        analog "key" columns instead of a clean categorical, so trees
        lose their free split-on-role shortcut too. Use this if the
        explicit-role variants don't separate models enough — it's a
        harder, less clean-hypothesis version.

Usage:
    python main.py --dataset feature_role_switching --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
    python -m datasets.feature_role_switching
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

N_TRAIN = 1000
N_TEST = 200
MASTER_SEED_OFFSET = 600


def _make_role_switch(
    n: int,
    n_roles: int,
    pool_size: int,
    rng,
    far_split: bool = False,
    explicit_role: bool = True,
    latent_bits: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n_signal_cols = n_roles * 2
    assert pool_size >= n_signal_cols, "pool_size must fit 2 signal cols per role"

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
    score = a * b + rng.normal(0, 0.5, n)  # mild label noise, matches context_length.py style
    y = (score > 0).astype(int)

    feature_blocks = [X_pool]

    if explicit_role:
        feature_blocks.append(role.reshape(-1, 1).astype(float))

    if latent_bits > 0:
        bits = (role[:, None] >> np.arange(latent_bits)) & 1
        key_cols = np.where(bits == 1, 1.0, -1.0) + rng.normal(0, 0.3, (n, latent_bits))
        feature_blocks.append(key_cols)

    X = np.hstack(feature_blocks)
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets = {}

    for n_roles in (2, 5, 10, 20):
        pool_size = max(n_roles * 2, 20)
        X, y = _make_role_switch(n_total, n_roles, pool_size, rng, far_split=False, explicit_role=True)
        datasets[f"role_switch_{n_roles}roles"] = (X, y)

    X, y = _make_role_switch(n_total, 10, 40, rng, far_split=True, explicit_role=True)
    datasets["role_switch_10roles_farsplit"] = (X, y)

    X, y = _make_role_switch(n_total, 10, 40, rng, far_split=False, explicit_role=False, latent_bits=4)
    datasets["role_switch_10roles_latent"] = (X, y)

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
    print("Running Feature Role Switching Benchmark")
    print("=" * 70)
    run_role_switch_check(model_names=models_to_run)
