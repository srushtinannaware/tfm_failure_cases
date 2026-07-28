"""
Generates the tweaked_dataset/ variants (noisy, irrelevant_features,
imbalanced) FROM the freshly-regenerated original_dataset/ base CSVs
(F1/F2/R1/R2/S1/S2 at 1000 train / 200 test / [200 ood_test]).

Run this AFTER original_dataset/generate_datasets.py has been re-run with
the reduced N_TRAIN/N_TEST/N_OOD_TEST values, so these tweaked variants
match the same row counts.

Variant naming matches kate_datasets.py exactly - no code changes needed
there for the standard variants:
    F1_noisy.csv, F1_irrelevant_features.csv
    F2_noisy.csv, F2_irrelevant_features.csv
    R1_noisy.csv, R1_imbalanced.csv
    R2_noisy.csv, R2_imbalanced.csv
    S1_noisy.csv, S1_irrelevant_features.csv
    S2_noisy.csv, S2_irrelevant_features.csv

PLUS one new variant not in the original set - see make_low_margin_variant()
docstring below for why, and the one kate_datasets.py edit needed to wire
it in if you want to use it.

Run from:
    /project/dl2026s/pereirak/tfm_failure_cases/datasets/tweaked_dataset
    python3 generate_tweaked_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).parent
ORIGINAL_DIR = THIS_DIR.parent / "original_dataset"
OUT_DIR = THIS_DIR

MASTER_SEED = 123  # separate seed stream from generate_datasets.py's MASTER_SEED=42

N_IRRELEVANT_COLS = 15
IMBALANCED_TARGET_POSITIVE_RATE = 0.10  # push toward 90:10 from R1/R2's ~25:75


def _load_base(name: str) -> pd.DataFrame:
    path = ORIGINAL_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run original_dataset/generate_datasets.py "
            "first so the base CSVs exist at the reduced row counts."
        )
    return pd.read_csv(path)


def _add_feature_noise(df: pd.DataFrame, feature_cols: list[str], sigma: float, rng) -> pd.DataFrame:
    """Adds Gaussian measurement noise to informative features only,
    leaving label and split untouched. This makes the true underlying
    rule harder to recover from observed values - a distinct failure
    mode from adding irrelevant columns."""
    noisy = df.copy()
    for col in feature_cols:
        noisy[col] = noisy[col] + rng.normal(0, sigma, size=len(noisy))
    return noisy


def _add_irrelevant_columns(df: pd.DataFrame, n_cols: int, rng, low: float = 0.0, high: float = 10.0) -> pd.DataFrame:
    """Appends pure-noise columns with zero relationship to the label,
    inserted before 'label'/'split' so those stay the last two columns."""
    result = df.copy()
    label = result.pop("label")
    split = result.pop("split")
    for i in range(n_cols):
        result[f"irrelevant_{i+1}"] = rng.uniform(low, high, size=len(result))
    result["label"] = label
    result["split"] = split
    return result


def _make_imbalanced(df: pd.DataFrame, target_positive_rate: float, rng) -> pd.DataFrame:
    """Undersamples the positive class WITHIN each split independently
    (so train and test each hit the target ratio on their own, and
    main.py's stratified re-split later preserves it). Uses undersampling
    rather than oversampling-with-duplicates, since duplicate rows are
    especially misleading for in-context-learning models like
    TabPFN/TabICL that condition directly on the training rows shown."""
    pieces = []
    for split_name, split_df in df.groupby("split"):
        n_pos = int((split_df["label"] == 1).sum())
        n_neg = int((split_df["label"] == 0).sum())
        target_pos = int(round(n_neg * target_positive_rate / (1 - target_positive_rate)))
        keep_pos = min(n_pos, target_pos)

        pos_rows = split_df[split_df["label"] == 1]
        neg_rows = split_df[split_df["label"] == 0]
        sampled_pos = pos_rows.sample(n=keep_pos, random_state=rng.integers(1_000_000))

        combined = pd.concat([neg_rows, sampled_pos], ignore_index=True)
        pieces.append(combined)

    result = pd.concat(pieces, ignore_index=True)
    return result.sample(frac=1, random_state=rng.integers(1_000_000)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# F1 / F2 - Feature Interaction: noisy + irrelevant_features
# ---------------------------------------------------------------------------
def make_F1_F2_variants() -> None:
    rng = np.random.default_rng(MASTER_SEED + 1)
    for name, feature_cols in (("F1", ["A", "B", "C"]), ("F2", ["A", "B", "C", "D"])):
        df = _load_base(name)

        noisy = _add_feature_noise(df, feature_cols, sigma=1.3, rng=rng)
        noisy.to_csv(OUT_DIR / f"{name}_noisy.csv", index=False)
        print(f"Wrote {name}_noisy.csv - {len(noisy)} rows")

        irrelevant = _add_irrelevant_columns(df, N_IRRELEVANT_COLS, rng=rng)
        irrelevant.to_csv(OUT_DIR / f"{name}_irrelevant_features.csv", index=False)
        print(f"Wrote {name}_irrelevant_features.csv - {len(irrelevant)} rows, "
              f"{N_IRRELEVANT_COLS} irrelevant cols added")


# ---------------------------------------------------------------------------
# S1 / S2 - Causal: noisy + irrelevant_features (ood_test rows get the SAME
# treatment as train/test, so the causal-shortcut check still has all 3
# splits to compare afterward).
# ---------------------------------------------------------------------------
def make_S1_S2_variants() -> None:
    rng = np.random.default_rng(MASTER_SEED + 2)
    causal_features = {
        "S1": ["A", "Spur_shortcut", "B"],
        "S2": ["A", "Spur1_shortcut", "B", "Spur2_shortcut", "C"],
    }
    for name, feature_cols in causal_features.items():
        df = _load_base(name)

        noisy = _add_feature_noise(df, feature_cols, sigma=0.3, rng=rng)
        noisy.to_csv(OUT_DIR / f"{name}_noisy.csv", index=False)
        print(f"Wrote {name}_noisy.csv - {len(noisy)} rows "
              f"(splits: {noisy['split'].value_counts().to_dict()})")

        irrelevant = _add_irrelevant_columns(df, N_IRRELEVANT_COLS, rng=rng, low=-3.0, high=3.0)
        irrelevant.to_csv(OUT_DIR / f"{name}_irrelevant_features.csv", index=False)
        print(f"Wrote {name}_irrelevant_features.csv - {len(irrelevant)} rows, "
              f"{N_IRRELEVANT_COLS} irrelevant cols added "
              f"(splits: {irrelevant['split'].value_counts().to_dict()})")


# ---------------------------------------------------------------------------
# R1 / R2 - Robustness: noisy (informative features only, existing noise_*
# columns left alone) + imbalanced (undersampled toward 90:10)
# ---------------------------------------------------------------------------
def make_R1_R2_variants() -> None:
    rng = np.random.default_rng(MASTER_SEED + 3)
    robustness_features = {
        "R1": ["A", "B", "C", "D"],
        "R2": ["A", "B", "C", "D", "E"],
    }
    for name, feature_cols in robustness_features.items():
        df = _load_base(name)

        noisy = _add_feature_noise(df, feature_cols, sigma=1.0, rng=rng)
        noisy.to_csv(OUT_DIR / f"{name}_noisy.csv", index=False)
        print(f"Wrote {name}_noisy.csv - {len(noisy)} rows")

        imbalanced = _make_imbalanced(df, IMBALANCED_TARGET_POSITIVE_RATE, rng=rng)
        imbalanced.to_csv(OUT_DIR / f"{name}_imbalanced.csv", index=False)
        actual_rate = imbalanced["label"].mean()
        print(f"Wrote {name}_imbalanced.csv - {len(imbalanced)} rows "
              f"(down from {len(df)} due to undersampling), "
              f"positive_rate={actual_rate:.4f} "
              f"(splits: {imbalanced['split'].value_counts().to_dict()})")


# ---------------------------------------------------------------------------
# NEW: low-margin variant for F1/F2 - a distinct failure case beyond the
# original three.
#
# WHY: F1/F2's threshold T is calibrated for ~50% positive rate, but most
# points sit comfortably far from T. A model can get high accuracy by
# learning a rough separating region without actually recovering the exact
# interaction rule. This variant resamples toward points whose raw score
# sits close to T (small margin), so getting them right requires the model
# to have actually learned the precise rule, not just the rough boundary -
# this directly tests calibration/precision near the decision boundary,
# a different failure axis than noise or irrelevant features.
#
# NOTE: this is a NEW variant name not in kate_datasets.py's
# SIMPLE_VARIANT_FILES dict. To actually use it in a benchmark run, you'd
# add one line there yourself, e.g.:
#     "F1_low_margin": (TWEAKED_DIR, "F1_low_margin.csv"),
# I'm not editing kate_datasets.py myself here since that's shared team
# code - your call whether to wire it in.
# ---------------------------------------------------------------------------
def make_low_margin_variant(name: str, feature_cols: list[str], score_fn, margin_width: float) -> None:
    df = _load_base(name)
    scores = score_fn(df)
    T = df.attrs.get("threshold_T")  # not stored by default; see note below

    # Since threshold_T isn't saved in the CSV itself, approximate it as the
    # score value separating label 0 from label 1 in this data (median of
    # the two classes' boundary region) - good enough for a resampling
    # filter, not meant to exactly recover generate_datasets.py's T.
    pos_scores = scores[df["label"] == 1]
    neg_scores = scores[df["label"] == 0]
    approx_T = (pos_scores.min() + neg_scores.max()) / 2

    near_boundary_mask = (scores - approx_T).abs() < margin_width
    filtered = df[near_boundary_mask].copy()

    filtered.to_csv(OUT_DIR / f"{name}_low_margin.csv", index=False)
    print(f"Wrote {name}_low_margin.csv - {len(filtered)} rows "
          f"(kept rows within +/-{margin_width} of approx threshold {approx_T:.2f}), "
          f"splits: {filtered['split'].value_counts().to_dict()}")


def make_low_margin_variants() -> None:
    make_low_margin_variant("F1", ["A", "B", "C"], lambda df: df["A"] * df["B"] + df["C"], margin_width=3.0)
    make_low_margin_variant("F2", ["A", "B", "C", "D"], lambda df: df["A"] * df["B"] * df["C"] - df["D"], margin_width=8.0)


if __name__ == "__main__":
    print("=" * 70)
    print("F1 / F2 - noisy + irrelevant_features")
    print("=" * 70)
    make_F1_F2_variants()

    print()
    print("=" * 70)
    print("S1 / S2 - noisy + irrelevant_features")
    print("=" * 70)
    make_S1_S2_variants()

    print()
    print("=" * 70)
    print("R1 / R2 - noisy + imbalanced")
    print("=" * 70)
    make_R1_R2_variants()

    print()
    print("=" * 70)
    print("NEW: F1 / F2 low_margin (boundary-precision stress test)")
    print("=" * 70)
    make_low_margin_variants()