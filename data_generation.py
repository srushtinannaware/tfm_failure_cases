"""
Synthetic "wide, short" tabular dataset generator.

The point of this generator is to reproduce the regime where tabular
foundation models built on attention/in-context-learning are reported to
struggle: very few rows relative to a large number of columns (e.g. 100
rows x 5,000 columns), with predictive signal genuinely spread across a
meaningful fraction of those columns (not just the first few) -- similar
to real-world wide data such as gene-expression panels or wide sensor
arrays.

If the informative signal were concentrated in a handful of columns,
per-estimator feature subsampling (used internally by TabPFN once the
feature count exceeds its per-estimator budget, see models.py) would
barely matter, because any random slice of features would probably still
catch a couple of the informative ones. Scaling the number of informative
columns roughly with total column count is what actually stresses that
architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification, make_regression


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_train: int
    n_test: int
    task: str
    seed: int


def _n_informative(n_features: int, frac: float = 0.10, min_k: int = 8, max_k: int = 300) -> int:
    """Scale informative-feature count with total column count (capped)."""
    k = int(round(n_features * frac))
    k = max(min_k, min(max_k, k))
    return min(k, n_features)


def make_wide_short_classification(
    n_features: int,
    n_train: int = 100,
    n_test: int = 500,
    seed: int = 0,
    n_classes: int = 2,
) -> Dataset:
    n_total = n_train + n_test
    n_informative = _n_informative(n_features)
    # n_redundant adds linear combinations of informative features (extra
    # "noise-that-correlates-with-signal" columns, common in real wide data)
    n_redundant = min(n_informative, n_features - n_informative)

    X, y = make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.03,
        class_sep=1.0,
        shuffle=True,
        random_state=seed,
    )
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]

    return Dataset(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:n_total],
        y_test=y[n_train:n_total],
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
        task="classification",
        seed=seed,
    )


def make_wide_short_regression(
    n_features: int,
    n_train: int = 100,
    n_test: int = 500,
    seed: int = 0,
) -> Dataset:
    n_total = n_train + n_test
    n_informative = _n_informative(n_features)

    X, y = make_regression(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_informative,
        noise=10.0,
        shuffle=True,
        random_state=seed,
    )
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]

    return Dataset(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:n_total],
        y_test=y[n_train:n_total],
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
        task="regression",
        seed=seed,
    )


def make_dataset(task: str, n_features: int, n_train: int = 100, n_test: int = 500, seed: int = 0) -> Dataset:
    if task == "classification":
        return make_wide_short_classification(n_features, n_train, n_test, seed)
    elif task == "regression":
        return make_wide_short_regression(n_features, n_train, n_test, seed)
    raise ValueError(f"Unknown task: {task}")


# Default column-count sweep for the failure-case study.
# 100 rows stays fixed; columns grow from a regime every model handles
# comfortably up through 2,000, where the accuracy and predict-time
# divergence between models is already clear. 5,000 columns is no longer
# the default (the marginal signal past 2,000 doesn't justify the extra
# runtime) but remains available as a spot check via
# `--column-sweep ... 5000`.
COLUMN_SWEEP = [50, 100, 200, 500, 1000, 2000]

if __name__ == "__main__":
    for n_feat in COLUMN_SWEEP:
        ds = make_dataset("classification", n_feat, seed=0)
        print(
            f"n_features={n_feat:>5}  n_informative~{_n_informative(n_feat):>4}  "
            f"X_train={ds.X_train.shape}  X_test={ds.X_test.shape}"
        )
