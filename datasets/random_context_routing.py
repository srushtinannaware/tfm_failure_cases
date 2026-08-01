"""
Random Context Routing Dataset — fine-tuned to isolate TabPFN v3 failure
while TabICL v2 and CatBoost remain relatively robust.

Key insight:
    TabPFN v3 has a hard context window limit — beyond ~3000-5000 training
    rows its attention degrades sharply (quadratic memory).
    TabICL v2 uses a different retrieval mechanism that scales better.
    CatBoost with cat_features gets native per-category statistics —
    it degrades gracefully at moderate M (many keys) because it computes
    exact per-key target means, not soft attention over all rows.

Two levers to separate TabPFN v3 from TabICL v2:

LEVER 1 — Large N_TRAIN (context size stress):
    Push N_TRAIN to 3000-5000. TabPFN v3 starts subsampling internally
    or degrading due to quadratic attention cost. TabICL v2 handles
    this better. CatBoost is unaffected by row count beyond a point.

LEVER 2 — Key encoding as FLOAT not INT:
    TabPFN v3 embeds features through a learned column encoder tuned
    on smooth continuous priors. A key presented as a raw float integer
    (e.g. key=147.0) lands far outside that prior's typical range
    when M is large. TabICL v2's column-grouping mechanism is less
    sensitive to this because it normalizes per-feature internally.
    CatBoost with cat_features bypasses this entirely — it never uses
    the raw float value, only the category identity.

LEVER 3 — Key collision zone (M just above N_TRAIN/2):
    At this M, most keys appear exactly once or twice in training.
    TabPFN v3's soft attention spreads probability mass over ALL
    training rows when looking up a key, so rare keys get drowned out
    by the many other rows. TabICL v2's retrieval is sharper.
    CatBoost computes exact per-key statistics — one training example
    is enough to get a perfect prediction for that key.

Usage:
    python main.py --dataset random_context_routing \
        --models catboost tabpfn_v2 tabpfn_v3 tabicl_v2
    python -m datasets.random_context_routing
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

# ── key constants ─────────────────────────────────────────────────────────
# LEVER 1: push N_TRAIN high enough to stress TabPFN v3's context window
# TabPFN v3 starts degrading noticeably above ~3000 rows
N_TRAIN          = 3000
N_TEST           = 400
N_DECOY_FEATURES = 10
MASTER_SEED_OFFSET = 800
LABEL_FLIP_PROB    = 0.02
KEY_COL_IDX        = 0   # column 0 is always the key


def _make_routing(
    n: int,
    n_keys: int,
    n_decoy: int,
    rng,
    float_key: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    float_key=True  → key stored as raw float (e.g. 147.0)
                       stresses TabPFN v3's continuous feature encoder
    float_key=False → key stored as small integer (0..M-1)
                       baseline condition
    """
    keys   = rng.integers(0, n_keys, size=n)
    lookup = rng.integers(0, 2, size=n_keys)
    y      = lookup[keys]

    flips  = rng.random(n) < LABEL_FLIP_PROB
    y      = np.where(flips, 1 - y, y)

    decoys = rng.normal(0, 1, size=(n, n_decoy))

    if float_key:
        # LEVER 2: encode key as large float — outside TabPFN v3's prior range
        # key=147.0 looks like a continuous feature to TabPFN's encoder
        # CatBoost with cat_features ignores the raw value entirely
        key_col = keys.astype(float).reshape(-1, 1)
    else:
        # baseline: small integer, less distribution-shift stress
        key_col = keys.reshape(-1, 1).astype(float)

    X = np.hstack([key_col, decoys])
    return X, y


def _make_catboost_with_cat():
    """CatBoost with explicit categorical key — uses per-key target stats."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        random_state=42,
        verbose=False,
        cat_features=[KEY_COL_IDX],
    )


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng     = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── original variants (baseline, unchanged) ───────────────────────────
    for n_keys in (10, 50, 200, 800, 2000):
        X, y = _make_routing(n_total, n_keys, N_DECOY_FEATURES, rng,
                             float_key=False)
        datasets[f"routing_M{n_keys}"] = (X, y)

    # ── LEVER 2: float key encoding — stresses TabPFN v3 specifically ─────
    # Same M values but key stored as raw float
    # Expected: TabPFN v3 degrades faster than TabICL v2 and CatBoost
    for n_keys in (50, 200, 500):
        X, y = _make_routing(n_total, n_keys, N_DECOY_FEATURES, rng,
                             float_key=True)
        datasets[f"routing_M{n_keys}_floatkey"] = (X, y)

    # ── LEVER 3: collision zone — M just above N_TRAIN//2 ─────────────────
    # Most keys appear exactly 1-2 times in training
    # TabPFN v3 soft attention gets diluted by all other rows
    # TabICL v2 sharper retrieval should handle this better
    # CatBoost exact per-key stats: 1 example = perfect prediction
    for n_keys in (1000, 1500, 2000, 3000):
        X, y = _make_routing(n_total, n_keys, N_DECOY_FEATURES, rng,
                             float_key=False)
        datasets[f"routing_collision_M{n_keys}"] = (X, y)

    # ── LEVER 1+2 combined: large float keys ──────────────────────────────
    # Both context stress AND encoding stress at once
    # Should maximally separate TabPFN v3 from TabICL v2
    for n_keys in (200, 500, 1000):
        X, y = _make_routing(n_total, n_keys, N_DECOY_FEATURES, rng,
                             float_key=True)
        datasets[f"routing_combined_M{n_keys}"] = (X, y)

    return datasets


def run_context_routing_check(
    model_names: list[str],
    seeds: list[int] = (0, 1),
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
            X_test,  y_test  = X[N_TRAIN:], y[N_TRAIN:]

            # compute shots per key from variant name
            try:
                m_part    = [p for p in variant_name.split("_") if p.startswith("M")][0]
                n_keys    = int(m_part[1:])
                shots     = round(N_TRAIN / n_keys, 2)
            except (IndexError, ValueError):
                n_keys, shots = -1, -1

            for model_name, factory in model_factories.items():
                print(
                    f"[{variant_name}] {model_name} "
                    f"(seed={seed}, M={n_keys}, shots/key={shots})..."
                )

                base_row = {
                    "dataset_variant" : variant_name,
                    "model"           : model_name,
                    "seed"            : seed,
                    "n_keys"          : n_keys,
                    "shots_per_key"   : shots,
                    "n_train"         : N_TRAIN,
                    "n_test"          : N_TEST,
                    "n_features"      : int(X.shape[1]),
                }

                try:
                    # give CatBoost categorical treatment of the key column
                    if "catboost" in model_name.lower():
                        model = _make_catboost_with_cat()
                    else:
                        model = factory()

                    metrics = evaluate_classifier(
                        model   = model,
                        X_train = X_train,
                        X_test  = X_test,
                        y_train = y_train,
                        y_test  = y_test,
                    )
                    row = {**base_row, **metrics, "status": "success", "error": ""}
                    print(
                        f"  Accuracy={metrics['accuracy']:.4f} | "
                        f"F1={metrics['f1_macro']:.4f} | "
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
        for col in row:
            if col not in all_columns:
                all_columns.append(col)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results -> {csv_path}")
    return rows


if __name__ == "__main__":
    # RealMLP removed, 2 seeds only
    models_to_run = ["catboost", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]

    print("=" * 70)
    print("Running Random Context Routing — TabPFN v3 isolation experiment")
    print("=" * 70)
    run_context_routing_check(model_names=models_to_run)