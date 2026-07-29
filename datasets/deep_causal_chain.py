"""
Deep compositional causal chain: A -> B -> C -> D -> E -> label (5 hops,
vs. S1's 1 hop and S2's 2 hops). Tests whether models can trace a LONGER
causal dependency chain rather than falling back on a cheap shortcut
feature tied to the EARLY node - directly related to the "Faith and Fate"
compositional-reasoning-limits literature: in-context foundation models
may have priors trained on shallower compositional structure, so a longer
chain could specifically expose that limit, while tree/MLP baselines
(which fit the function directly rather than via an in-context prior)
may be less affected.

Same design pattern as kate_datasets.py's S1/S2:
  - train/test: shortcut correlates with the true causal path (rho=+0.9)
  - ood_test:   that correlation FLIPS (rho=-0.9)
A model that actually traced the 5-hop chain (A->B->C->D->E) keeps working
on ood_test; a model that leaned on the 1-hop shortcut degrades sharply -
and a longer chain should make that shortcut MORE tempting to lean on,
since actually tracing 5 hops is harder than tracing S1/S2's 1-2 hops.

Usage:
    # ordinary in-distribution accuracy (train+test combined, like
    # kate_datasets.py's get_datasets does for S1/S2):
    python main.py --dataset deep_causal_chain --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

    # the REAL test - test vs ood_test drop, bypasses main.py's auto-split:
    python -m datasets.deep_causal_chain
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path

import numpy as np
import pandas as pd

N_TRAIN = 1000
N_TEST = 200
N_OOD_TEST = 200
MASTER_SEED_OFFSET = 100  # separate stream from kate_datasets.py's generation

RHO = 0.9
HOP_NOISE_SIGMA = 0.4
LABEL_NOISE_SIGMA = 0.3
HOP_WEIGHT = 1.2  # signal strength carried at each hop; <1 would decay faster

FEATURE_COLUMNS = ["A", "Spur_shortcut", "B", "C", "D", "E"]


def _generate_chain(n: int, shortcut_sign: int, rng) -> dict[str, np.ndarray]:
    A = rng.normal(0, 1, n)
    Spur = shortcut_sign * RHO * A + np.sqrt(1 - RHO ** 2) * rng.normal(0, 1, n)

    B = HOP_WEIGHT * A + rng.normal(0, HOP_NOISE_SIGMA, n)
    C = HOP_WEIGHT * B + rng.normal(0, HOP_NOISE_SIGMA, n)
    D = HOP_WEIGHT * C + rng.normal(0, HOP_NOISE_SIGMA, n)
    E = HOP_WEIGHT * D + rng.normal(0, HOP_NOISE_SIGMA, n)

    logit_noise = rng.normal(0, LABEL_NOISE_SIGMA, n)
    label = (1.0 * E + logit_noise > 0).astype(int)

    return {"A": A, "Spur_shortcut": Spur, "B": B, "C": C, "D": D, "E": E, "label": label}


def generate_full(seed: int) -> pd.DataFrame:
    """Returns one DataFrame with train/test/ood_test rows and a 'split'
    column, matching kate_datasets.py's S1/S2 CSV format exactly."""
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)

    train_data = _generate_chain(N_TRAIN, shortcut_sign=+1, rng=rng)
    test_data = _generate_chain(N_TEST, shortcut_sign=+1, rng=rng)
    ood_data = _generate_chain(N_OOD_TEST, shortcut_sign=-1, rng=rng)  # shortcut flips here

    rows = []
    for data, split_name in ((train_data, "train"), (test_data, "test"), (ood_data, "ood_test")):
        df = pd.DataFrame({col: data[col] for col in FEATURE_COLUMNS})
        df["label"] = data["label"]
        df["split"] = split_name
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def save_csv(seed: int = 0) -> Path:
    """Optional: save a CSV copy for inspection/reproducibility, mirroring
    original_dataset/'s files. Not required for get_datasets() or
    run_causal_shortcut_check() below, which regenerate deterministically
    from the seed instead."""
    out_dir = Path(__file__).parent / "deep_chain_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_full(seed)
    out_path = out_dir / "deep_causal_chain.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} - {len(df)} rows, splits: {df['split'].value_counts().to_dict()}")
    return out_path


# ---------------------------------------------------------------------------
# main.py entry point: ordinary in-distribution run (train+test combined,
# ood_test dropped) - same convention as kate_datasets.py's S1/S2 handling.
# ---------------------------------------------------------------------------
def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns BOTH variants for comparison:
      - deep_causal_chain: ORIGINAL version, B/C/D/E all exposed as
        features. Kept as a valid negative-finding baseline - since the
        immediate parent of label (E) is directly observable, chain depth
        turned out not to matter; models never needed to trace anything.
      - deep_causal_chain_hidden: FIXED version, only A and Spur_shortcut
        exposed. B/C/D/E are hidden latent variables used only to generate
        the label. This forces a real choice: trust A's weak, noise-decayed
        marginal correlation with label (the true but indirect cause) vs.
        Spur_shortcut's clean rho=0.9 correlation with A (the shortcut).
        As chain depth grows, accumulated hop noise weakens A's signal,
        which should make the shortcut relatively MORE tempting - this is
        the actual mechanism the "depth increases shortcut reliance"
        hypothesis needs, which the original exposed version didn't test.
    """
    df = generate_full(seed)
    in_dist = df[df["split"] != "ood_test"]

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    exposed = in_dist.drop(columns=["split"]).copy()
    y_exposed = exposed.pop("label").to_numpy()
    datasets["deep_causal_chain"] = (exposed.to_numpy(), y_exposed)

    hidden = in_dist[["A", "Spur_shortcut", "label"]].copy()
    y_hidden = hidden.pop("label").to_numpy()
    datasets["deep_causal_chain_hidden"] = (hidden.to_numpy(), y_hidden)

    return datasets


# ---------------------------------------------------------------------------
# Dedicated causal-shortcut check (NOT run through main.py) - same pattern
# as kate_datasets.py's run_causal_shortcut_check().
# ---------------------------------------------------------------------------
def run_causal_shortcut_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
    expose_intermediates: bool = False,
) -> list[dict]:
    """Set expose_intermediates=True to reproduce the ORIGINAL (flawed)
    version for comparison. Default False runs the FIXED hidden-feature
    version - only A and Spur_shortcut as features - which is the real
    test of whether chain depth increases shortcut reliance."""
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []
    variant_label = "deep_causal_chain" if expose_intermediates else "deep_causal_chain_hidden"

    for seed in seeds:
        df = generate_full(seed)

        feature_cols = FEATURE_COLUMNS if expose_intermediates else ["A", "Spur_shortcut"]

        def _split_xy(split_name: str):
            sub = df[df["split"] == split_name]
            X = sub[feature_cols].to_numpy()
            y = sub["label"].to_numpy()
            return X, y

        X_train, y_train = _split_xy("train")
        X_test, y_test = _split_xy("test")
        X_ood, y_ood = _split_xy("ood_test")

        for model_name, factory in model_factories.items():
            print(f"[{variant_label}] {model_name} (seed={seed})...")

            base_row = {"variant": variant_label, "model": model_name, "seed": seed}

            try:
                metrics_test = evaluate_classifier(
                    model=factory(), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                )
                metrics_ood = evaluate_classifier(
                    model=factory(), X_train=X_train, X_test=X_ood, y_train=y_train, y_test=y_ood,
                )

                row = {**base_row, "status": "success", "error": ""}
                for key, value in metrics_test.items():
                    row[f"{key}_test"] = value
                for key, value in metrics_ood.items():
                    row[f"{key}_ood"] = value

                row["accuracy_drop"] = metrics_test["accuracy"] - metrics_ood["accuracy"]
                auc_test = metrics_test.get("roc_auc", np.nan)
                auc_ood = metrics_ood.get("roc_auc", np.nan)
                row["roc_auc_drop"] = (
                    np.nan if (np.isnan(auc_test) or np.isnan(auc_ood)) else auc_test - auc_ood
                )

                print(f"  test accuracy={metrics_test['accuracy']:.4f} | "
                      f"ood_test accuracy={metrics_ood['accuracy']:.4f} | "
                      f"drop={row['accuracy_drop']:.4f}")

            except Exception as error:
                row = {**base_row, "status": "failed", "error": str(error)}
                print(f"  Failed: {error}")

            rows.append(row)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{variant_label}_check_results.csv"

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
    save_csv(seed=0)  # optional inspection copy

    models_to_run = ["catboost", "realmlp", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]

    print("=" * 70)
    print("HIDDEN version (the real test) - only A, Spur_shortcut exposed")
    print("=" * 70)
    run_causal_shortcut_check(model_names=models_to_run, expose_intermediates=False)

    print()
    print("=" * 70)
    print("EXPOSED version (original baseline) - B/C/D/E all visible")
    print("=" * 70)
    run_causal_shortcut_check(model_names=models_to_run, expose_intermediates=True)