"""
Many-Class Classification Failure:
Tests model performance as number of output classes scales beyond the
training distribution of Tabular Foundation Models.

Key insight:
    TabPFN v2 was trained ONLY on datasets with up to 10 classes.
    TabPFN v3 extended this, but both degrade sharply as class count grows.
    CatBoost and RealMLP have NO such architectural limit — they grow
    naturally with more classes (one tree per class, more output neurons).

Design:
    Task is a Voronoi partition — each point is assigned to its nearest
    centroid. The rule is simple and consistent across all variants.
    ONLY the number of classes changes between variants — nothing else.
    This isolates class count as the single variable being tested.

Variants:
    many_class_2:   2  classes  (easy baseline, all models should ace this)
    many_class_5:   5  classes  (still within TabPFN v2 training range)
    many_class_10:  10 classes  (TabPFN v2 upper limit — edge case)
    many_class_15:  15 classes  (beyond TabPFN v2 — degradation expected)
    many_class_20:  20 classes  (well beyond limit — clear failure zone)
    many_class_25:  25 classes  (TabPFN v2 may crash or output garbage)

Usage 1 (Standard main.py harness):
    python main.py --dataset many_class \
        --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

Usage 2 (Standalone check):
    python -m datasets.many_class
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TOTAL        = 2000   # total samples per variant
N_FEATURES     = 10     # feature dimensionality (fixed across all variants)
CLASS_COUNTS   = [2, 5, 10, 15, 20, 25]
MASTER_SEED_OFFSET = 600


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _generate_voronoi(
    n: int,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Voronoi classification dataset.

    Each sample is assigned the label of its nearest centroid.
    Centroids are placed randomly in feature space with enough
    spread (scale=2) to avoid heavy overlap between classes.

    Parameters
    ----------
    n         : number of samples
    n_classes : number of Voronoi regions / class labels
    rng       : numpy random generator (for reproducibility)

    Returns
    -------
    X : shape (n, N_FEATURES)
    y : shape (n,)  integer labels in [0, n_classes)
    """
    # place centroids spread across feature space
    centroids = rng.normal(loc=0.0, scale=2.0, size=(n_classes, N_FEATURES))

    # sample data points from a tighter distribution
    X = rng.normal(loc=0.0, scale=1.0, size=(n, N_FEATURES))

    # assign each point to nearest centroid (Euclidean distance)
    # shape: (n, n_classes)
    dists = np.linalg.norm(
        X[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2,
    )
    y = dists.argmin(axis=1).astype(int)

    return X, y


# ---------------------------------------------------------------------------
# main.py entry point
# ---------------------------------------------------------------------------
def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns one dataset variant per class count.
    Each variant uses the SAME seed and feature space — only n_classes
    differs, isolating class count as the experimental variable.
    """
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for n_classes in CLASS_COUNTS:
        X, y = _generate_voronoi(N_TOTAL, n_classes, rng)
        datasets[f"many_class_{n_classes:02d}"] = (X, y)

    return datasets


# ---------------------------------------------------------------------------
# Standalone runner — matches the pattern of deep_causal_chain.py
# ---------------------------------------------------------------------------
def run_many_class_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
    train_ratio: float = 0.8,
) -> list[dict]:
    """
    Trains each model on each variant and records accuracy, F1, ROC-AUC.
    Saves results to results/many_class_check_results.csv.

    Parameters
    ----------
    model_names : list of model keys from models.py registry
    seeds       : list of random seeds (results are averaged over seeds)
    train_ratio : fraction of N_TOTAL used for training
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    n_train = int(N_TOTAL * train_ratio)
    n_test  = N_TOTAL - n_train

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
            # extract n_classes from variant name (e.g. "many_class_10" → 10)
            n_classes = int(variant_name.split("_")[-1])

            X_train, y_train = X[:n_train], y[:n_train]
            X_test,  y_test  = X[n_train:], y[n_train:]

            for model_name, factory in model_factories.items():
                print(
                    f"[{variant_name}] {model_name} "
                    f"(seed={seed}, n_classes={n_classes})..."
                )

                base_row = {
                    "dataset_variant" : variant_name,
                    "n_classes"       : n_classes,
                    "model"           : model_name,
                    "seed"            : seed,
                    "n_train"         : n_train,
                    "n_test"          : n_test,
                    "n_features"      : N_FEATURES,
                }

                try:
                    metrics = evaluate_classifier(
                        model   = factory(),
                        X_train = X_train,
                        X_test  = X_test,
                        y_train = y_train,
                        y_test  = y_test,
                    )

                    row = {**base_row, **metrics,
                           "status": "success", "error": ""}

                    print(
                        f"  Accuracy={metrics['accuracy']:.4f} | "
                        f"F1={metrics['f1_macro']:.4f} | "
                        f"Time={metrics['total_seconds']:.2f}s"
                    )

                except Exception as error:
                    row = {**base_row,
                           "status": "failed", "error": str(error)}
                    print(f"  FAILED: {error}")

                rows.append(row)

    # ── save results ──────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "many_class_check_results.csv"

    all_columns: list[str] = []
    for row in rows:
        for col in row:
            if col not in all_columns:
                all_columns.append(col)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results → {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    models_to_run = [
        "catboost",
        "realmlp",
        "tabpfn_v2",
        "tabpfn_v3",
        "tabicl_v2",
    ]

    print("=" * 70)
    print("Running Many-Class Classification Benchmark")
    print("Hypothesis: TFMs degrade beyond their trained class count limit")
    print("            CatBoost / RealMLP have no such architectural limit")
    print("=" * 70)
    print()

    run_many_class_check(model_names=models_to_run)