"""Sparse categorical lookup benchmark.

Each key maps to a randomly sampled binary label. Increasing the number of
keys at a fixed training budget reduces the expected observations per key.
Ten Gaussian decoy features and a small label-flip probability prevent the
task from being a noiseless lookup.
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

N_TRAIN = 3000
N_TEST = 400
N_DECOY_FEATURES = 10
MASTER_SEED_OFFSET = 800
LABEL_FLIP_PROB = 0.02
KEY_COL_IDX = 0

# Keys remain well separated after jitter, so the lookup is solvable in
# principle while using a numeric rather than native categorical encoding.
FLOAT_KEY_SCALE = 1000.0
FLOAT_KEY_JITTER = 0.5


def _make_routing(
    n: int,
    n_keys: int,
    n_decoy: int,
    rng,
    float_key: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    keys = rng.integers(0, n_keys, size=n)
    lookup = rng.integers(0, 2, size=n_keys)
    y = lookup[keys]

    flips = rng.random(n) < LABEL_FLIP_PROB
    y = np.where(flips, 1 - y, y)

    decoys = rng.normal(0, 1, size=(n, n_decoy))

    if float_key:
        jitter = rng.uniform(-FLOAT_KEY_JITTER, FLOAT_KEY_JITTER, n)
        key_col = (keys.astype(float) * FLOAT_KEY_SCALE + jitter).reshape(-1, 1)
    else:
        key_col = keys.reshape(-1, 1).astype(float)

    X = np.hstack([key_col, decoys])
    return X, y


def _make_catboost_with_cat():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=42, verbose=False, cat_features=[KEY_COL_IDX])


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns variant_name -> (X, y) for main.py compatibility.
    n_keys metadata is stored in run_context_routing_check directly."""
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    # Bridge sweep: 20 observations per key down to approximately 2.6.
    for n_keys in range(150, 1200, 200):
        X, y = _make_routing(n_total, n_keys, N_DECOY_FEATURES, rng, float_key=True)
        datasets[f"routing_combined_M{n_keys}"] = (X, y)

    return datasets


def run_context_routing_check(model_names: list[str], seeds: list[int] = (0, 1)) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
            # extract n_keys from variant name for shots_per_key calculation
            try:
                n_keys = int(variant_name.split("_M")[-1])
            except ValueError:
                n_keys = -1
            shots_per_key = round(N_TRAIN / n_keys, 3)
            X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
            X_test, y_test = X[N_TRAIN:], y[N_TRAIN:]

            for model_name, factory in model_factories.items():
                print(
                    f"[{variant_name}] {model_name} "
                    f"(seed={seed}, n_keys={n_keys}, shots/key={shots_per_key})..."
                )

                base_row = {
                    "dataset_variant": variant_name,
                    "model": model_name,
                    "seed": seed,
                    "n_keys": n_keys,
                    "shots_per_key": shots_per_key,
                    "n_train": N_TRAIN,
                    "n_test": N_TEST,
                    "n_features": int(X.shape[1]),
                }

                try:
                    if "catboost" in model_name.lower():
                        model = _make_catboost_with_cat()
                    else:
                        model = factory()

                    metrics = evaluate_classifier(
                        model=model,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
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

    gap_rows = _compute_gap_summary(rows)
    gap_csv_path = results_dir / "random_context_routing_gap_summary.csv"
    if gap_rows:
        gap_columns = list(gap_rows[0].keys())
        with gap_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=gap_columns)
            writer.writeheader()
            writer.writerows(gap_rows)
        print(f"Saved gap summary -> {gap_csv_path}")

    return rows


def _compute_gap_summary(rows: list[dict]) -> list[dict]:
    """For each (dataset_variant, seed), compute
    gap = TabICL-v2 accuracy - max(accuracy of the other successful models).
    This is the single-line version of the "flowing" plot: gap vs
    shots_per_key. Skips rows where TabICL-v2 or all other models failed."""
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        key = (row["dataset_variant"], row["seed"])
        by_key.setdefault(key, {})[row["model"]] = row

    gap_rows = []
    for (variant, seed), model_rows in by_key.items():
        icl_row = next((r for name, r in model_rows.items() if "tabicl" in name.lower()), None)
        if icl_row is None:
            continue
        other_accs = [
            r["accuracy"] for name, r in model_rows.items() if "tabicl" not in name.lower()
        ]
        if not other_accs:
            continue
        gap_rows.append(
            {
                "dataset_variant": variant,
                "seed": seed,
                "n_keys": icl_row["n_keys"],
                "shots_per_key": icl_row["shots_per_key"],
                "tabicl_accuracy": icl_row["accuracy"],
                "best_other_accuracy": max(other_accs),
                "gap": icl_row["accuracy"] - max(other_accs),
            }
        )
    gap_rows.sort(key=lambda r: (r["dataset_variant"], r["seed"]))
    return gap_rows


if __name__ == "__main__":
    models_to_run = ["catboost", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]

    print("=" * 70)
    print("Running Random Context Routing — bridge sweep + gap summary")
    print("=" * 70)
    run_context_routing_check(model_names=models_to_run, seeds=(0, 1))
