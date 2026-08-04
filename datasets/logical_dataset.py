"""Synthetic logical-reasoning binary classification datasets."""

from __future__ import annotations

import numpy as np


LOGICAL_TASKS = {
    "L1": {"operator": "AND", "n_relevant_features": 2, "expected_rate": 0.25},
    "L2": {"operator": "OR", "n_relevant_features": 2, "expected_rate": 0.75},
    "L3": {"operator": "XOR", "n_relevant_features": 4, "expected_rate": 0.50},
}
DEFAULT_NOISE_FEATURES = (0, 4)
N_SAMPLES = 1000


def primitive_conditions(
    X_relevant: np.ndarray,
    dataset_variant: str,
) -> np.ndarray:
    """Recompute all primitive threshold conditions from original features."""
    if dataset_variant not in LOGICAL_TASKS:
        raise ValueError(
            f"Unknown logical variant {dataset_variant!r}; "
            f"expected one of {tuple(LOGICAL_TASKS)}."
        )
    X_relevant = np.asarray(X_relevant)
    expected_features = int(
        LOGICAL_TASKS[dataset_variant]["n_relevant_features"]
    )
    if X_relevant.ndim != 2 or X_relevant.shape[1] != expected_features:
        raise ValueError(
            f"{dataset_variant} requires exactly {expected_features} "
            "relevant features."
        )
    return (X_relevant > 0).astype(np.int8)


def recompute_target(
    X_relevant: np.ndarray,
    dataset_variant: str,
) -> np.ndarray:
    """Recompute a target directly from the original continuous features."""
    conditions = primitive_conditions(X_relevant, dataset_variant)
    if dataset_variant == "L1":
        return np.logical_and(conditions[:, 0], conditions[:, 1]).astype(
            np.int8
        )
    if dataset_variant == "L2":
        return np.logical_or(conditions[:, 0], conditions[:, 1]).astype(
            np.int8
        )
    return np.logical_xor.reduce(conditions, axis=1).astype(np.int8)


def generate_logical_dataset(
    dataset_variant: str,
    n_noise_features: int = 0,
    seed: int = 0,
    n_samples: int = N_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian inputs and an AND, OR, or multi-condition XOR target."""
    if dataset_variant not in LOGICAL_TASKS:
        raise ValueError(
            f"Unknown logical variant {dataset_variant!r}; "
            f"expected one of {tuple(LOGICAL_TASKS)}."
        )
    if (
        not isinstance(n_noise_features, (int, np.integer))
        or n_noise_features < 0
    ):
        raise ValueError("n_noise_features must be a non-negative integer.")
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    rng = np.random.default_rng(seed)
    n_relevant = int(
        LOGICAL_TASKS[dataset_variant]["n_relevant_features"]
    )
    X_relevant = rng.normal(size=(n_samples, n_relevant))
    y = recompute_target(X_relevant, dataset_variant)
    X_noise = rng.normal(size=(n_samples, n_noise_features))

    # X contains original continuous variables only. Primitive conditions and
    # all other engineered logical variables remain hidden from the models.
    X = np.hstack((X_relevant, X_noise))
    return X, y


def get_datasets(
    seed: int,
    mode: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return all logical/noise variants using the repository contract."""
    if mode not in (None, "all"):
        raise ValueError("logical_dataset supports only mode=None or 'all'.")
    return {
        f"{variant}_noise_{n_noise}": generate_logical_dataset(
            dataset_variant=variant,
            n_noise_features=n_noise,
            seed=seed,
        )
        for n_noise in DEFAULT_NOISE_FEATURES
        for variant in LOGICAL_TASKS
    }
