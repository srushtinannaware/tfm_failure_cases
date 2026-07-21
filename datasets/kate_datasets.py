"""
Kate's Feature Interaction / SCM (Causal) / Robustness datasets, ALL in one
module: F1, F2, S1, S2, R1, R2 and their noisy / irrelevant-features /
imbalanced variants.

Model definitions live only in models.py, as they should - this file never
constructs a model itself, it only loads data and (for the shortcut check)
calls get_model() from models.py to fetch the same factories main.py uses.

--------------------------------------------------------------------------
USAGE 1 - the shared team harness (ordinary in-distribution accuracy):

    python main.py --dataset kate_datasets --models catboost realmlp

This runs ALL 18 variants (F1/F2/R1/R2's 3 variants each, plus S1/S2's
train+test combined) through main.py's usual pipeline: it does its own
random train/test split per variant and saves one combined results CSV +
accuracy plot, exactly like any other --dataset value your teammates use.

--------------------------------------------------------------------------
USAGE 2 - the S1/S2 causal-shortcut robustness check (NOT the same thing):

    python -m datasets.kate_datasets

S1 and S2 are built with a spurious "shortcut" feature that correlates with
the true cause in normal data, but has that correlation FLIPPED in a
special ood_test split. main.py's automatic random splitting would shuffle
that flip away and silently destroy the test, so this check bypasses
main.py entirely: it trains once on the designed TRAIN split, then
evaluates the SAME model separately on TEST and OOD_TEST. A big
accuracy/AUC drop from TEST to OOD_TEST means that model leaned on the
spurious shortcut instead of the real causal chain - that drop is your
causal-reasoning failure signature.

Edit the model_names list in the __main__ block at the bottom to change
which models this runs.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path

import numpy as np
import pandas as pd

ORIGINAL_DIR = Path(__file__).parent / "original_dataset"
TWEAKED_DIR = Path(__file__).parent / "tweaked_dataset"


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------
def _read_csv(subfolder: Path, filename: str) -> pd.DataFrame:
    path = subfolder / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset file at {path}, but it doesn't exist. "
            f"Check that your CSVs are in datasets/original_dataset/ and "
            f"datasets/tweaked_dataset/."
        )
    return pd.read_csv(path)


def _load_simple_variant(subfolder: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """For F1 / F2 / R1 / R2 - style files (a plain train+test CSV)."""
    df = _read_csv(subfolder, filename).drop(columns=["split"])
    y = df.pop("label").to_numpy()
    X = df.to_numpy()
    return X, y


def _load_causal_splits(subfolder: Path, filename: str):
    """
    For S1 / S2 - style files with a designed train / test / ood_test split.
    Returns the three splits separately and unmodified - nothing here
    re-shuffles or re-combines them.

    Returns
    -------
    X_train, y_train, X_test, y_test, X_ood, y_ood
    """
    df = _read_csv(subfolder, filename)

    def _split_xy(sub_df: pd.DataFrame):
        sub_df = sub_df.drop(columns=["split"]).copy()
        y = sub_df.pop("label").to_numpy()
        X = sub_df.to_numpy()
        return X, y

    X_train, y_train = _split_xy(df[df["split"] == "train"])
    X_test, y_test = _split_xy(df[df["split"] == "test"])
    X_ood, y_ood = _split_xy(df[df["split"] == "ood_test"])
    return X_train, y_train, X_test, y_test, X_ood, y_ood


# ---------------------------------------------------------------------------
# One place to look if a filename ever changes
# ---------------------------------------------------------------------------
SIMPLE_VARIANT_FILES = {
    "F1_base": (ORIGINAL_DIR, "F1.csv"),
    "F1_noisy": (TWEAKED_DIR, "F1_noisy.csv"),
    "F1_irrelevant": (TWEAKED_DIR, "F1_irrelevant_features.csv"),
    "F2_base": (ORIGINAL_DIR, "F2.csv"),
    "F2_noisy": (TWEAKED_DIR, "F2_noisy.csv"),
    "F2_irrelevant": (TWEAKED_DIR, "F2_irrelevant_features.csv"),
    "R1_base": (ORIGINAL_DIR, "R1.csv"),
    "R1_noisy": (TWEAKED_DIR, "R1_noisy.csv"),
    "R1_imbalanced": (TWEAKED_DIR, "R1_imbalanced.csv"),
    "R2_base": (ORIGINAL_DIR, "R2.csv"),
    "R2_noisy": (TWEAKED_DIR, "R2_noisy.csv"),
    "R2_imbalanced": (TWEAKED_DIR, "R2_imbalanced.csv"),
}

CAUSAL_VARIANT_FILES = {
    "S1_base": (ORIGINAL_DIR, "S1.csv"),
    "S1_noisy": (TWEAKED_DIR, "S1_noisy.csv"),
    "S1_irrelevant": (TWEAKED_DIR, "S1_irrelevant_features.csv"),
    "S2_base": (ORIGINAL_DIR, "S2.csv"),
    "S2_noisy": (TWEAKED_DIR, "S2_noisy.csv"),
    "S2_irrelevant": (TWEAKED_DIR, "S2_irrelevant_features.csv"),
}


# ---------------------------------------------------------------------------
# main.py entry point - called automatically as
# datasets.kate_datasets.get_datasets(seed)
# ---------------------------------------------------------------------------
def get_datasets(seed):
    # `seed` is accepted for interface-compatibility with main.py (which
    # always calls get_datasets(seed)), but these are fixed, pre-generated
    # CSVs rather than regenerated per seed. main.py only uses `seed` for its
    # own train_test_split, so re-running with multiple seeds still gives you
    # different train/test partitions of the same underlying data.
    del seed

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for variant_name, (subfolder, filename) in SIMPLE_VARIANT_FILES.items():
        datasets[variant_name] = _load_simple_variant(subfolder, filename)

    # S1/S2: combine train+test (drop ood_test) so main.py's own random
    # split still works normally. This gives ordinary in-distribution
    # accuracy only - it does NOT test the causal-shortcut failure mode.
    # Use run_causal_shortcut_check() below for that.
    for variant_name, (subfolder, filename) in CAUSAL_VARIANT_FILES.items():
        X_train, y_train, X_test, y_test, _X_ood, _y_ood = _load_causal_splits(subfolder, filename)
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        datasets[variant_name] = (X, y)

    return datasets


# ---------------------------------------------------------------------------
# Dedicated S1/S2 causal-shortcut robustness check (NOT run through main.py)
# ---------------------------------------------------------------------------
def run_causal_shortcut_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
    variants: dict[str, str] | None = None,
) -> list[dict]:
    """
    For each S1/S2 variant: train on the TRAIN split, then evaluate the SAME
    (freshly re-fit) model separately on TEST and OOD_TEST. Saves a combined
    results CSV to results/causal_shortcut_check_results.csv.
    """
    from metrics import evaluate_classifier
    from models import get_model

    if variants is None:
        variants = CAUSAL_VARIANT_FILES

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for variant_name, (subfolder, filename) in variants.items():
        X_train, y_train, X_test, y_test, X_ood, y_ood = _load_causal_splits(subfolder, filename)

        for seed in seeds:
            for model_name, factory in model_factories.items():
                print(f"[{variant_name}] {model_name} (seed={seed})...")

                base_row = {
                    "dataset_variant": variant_name,
                    "model": model_name,
                    "seed": seed,
                }

                try:
                    # Fresh model instance per test set, so evaluating on
                    # ood_test can never be affected by anything left over
                    # from evaluating on test.
                    metrics_test = evaluate_classifier(
                        model=factory(),
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                    )
                    metrics_ood = evaluate_classifier(
                        model=factory(),
                        X_train=X_train, X_test=X_ood,
                        y_train=y_train, y_test=y_ood,
                    )

                    row = {**base_row, "status": "success", "error": ""}
                    for key, value in metrics_test.items():
                        row[f"{key}_test"] = value
                    for key, value in metrics_ood.items():
                        row[f"{key}_ood"] = value

                    row["accuracy_drop"] = (
                        metrics_test["accuracy"] - metrics_ood["accuracy"]
                    )
                    auc_test = metrics_test.get("roc_auc", np.nan)
                    auc_ood = metrics_ood.get("roc_auc", np.nan)
                    row["roc_auc_drop"] = (
                        np.nan if (np.isnan(auc_test) or np.isnan(auc_ood))
                        else auc_test - auc_ood
                    )

                    print(
                        f"  test accuracy={metrics_test['accuracy']:.4f} | "
                        f"ood_test accuracy={metrics_ood['accuracy']:.4f} | "
                        f"drop={row['accuracy_drop']:.4f}"
                    )

                except Exception as error:
                    row = {**base_row, "status": "failed", "error": str(error)}
                    print(f"  Failed: {error}")

                rows.append(row)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "causal_shortcut_check_results.csv"

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
    # Edit this list to change which of your models get run.
    run_causal_shortcut_check(model_names=["catboost", "realmlp"])