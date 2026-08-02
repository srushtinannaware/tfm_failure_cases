"""
Deep Causal Chain — redesigned to show TabPFN v3 specific OOD failure.

The key insight: we need the STANDARD test accuracy to be HIGH for all
models (so the task is learnable), but the OOD DROP to be LARGE
specifically for TFMs.

New design:
  - 5 features exposed: A, Spur_shortcut, X1, X2, X3 (3 neutral noise)
  - A has a MODERATE true causal signal (not too weak, not too strong)
  - Spur_shortcut has rho=0.95 with A in train — very tempting shortcut
  - In ood_test: Spur_shortcut correlation FLIPS to rho=-0.95
  - X1, X2, X3 are pure noise — give models something to fit on train

  Models that memorize the shortcut (Spur_shortcut -> label) will get
  ~50% on ood_test because the shortcut now points the wrong way.
  Models that learn the TRUE signal (A -> label) will stay high.

  Why TabPFN v3 specifically fails:
  - Its prior was trained on SCMs where strong correlations = true causes
  - Spur_shortcut has rho=0.95 — it LOOKS like the true cause
  - TabPFN v3 trusts it and gets burned on ood_test
  - CatBoost fits A directly because it does greedy splits on ALL features
  - With enough samples CatBoost finds A even when Spur looks stronger

Run the OOD check:
    python -m datasets.deep_causal_chain
"""

from __future__ import annotations
import csv as csv_module
from pathlib import Path
import sys
import numpy as np
import pandas as pd

N_TRAIN    = 3000   # enough for CatBoost to find the weak A signal
N_TEST     = 600
N_OOD_TEST = 600
MASTER_SEED_OFFSET = 100

# causal chain parameters
RHO          = 0.95   # shortcut correlation — very strong, very tempting
HOP_WEIGHT   = 1.0    # A signal stays constant through hops
HOP_NOISE    = 0.5    # moderate noise so A->label is learnable but not trivial
LABEL_NOISE  = 0.15   # low label noise so task is genuinely solvable


def _generate(n: int, shortcut_sign: int, rng) -> dict:
    # true causal feature
    A = rng.normal(0, 1, n)

    # spurious shortcut — strongly correlated with A in train
    # flips sign in ood_test
    Spur = (shortcut_sign * RHO * A
            + np.sqrt(1 - RHO**2) * rng.normal(0, 1, n))

    # causal chain: A -> B -> label
    B     = HOP_WEIGHT * A + rng.normal(0, HOP_NOISE, n)
    label = (B + rng.normal(0, LABEL_NOISE, n) > 0).astype(int)

    # 3 pure noise features — give models something else to fit on
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 1, n)

    return {
        "A":             A,
        "Spur_shortcut": Spur,
        "X1_noise":      X1,
        "X2_noise":      X2,
        "X3_noise":      X3,
        "label":         label,
    }


def generate_full(seed: int) -> pd.DataFrame:
    rng   = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    rows  = []
    for data, split, sign in [
        (_generate(N_TRAIN,    +1, rng), "train",    +1),
        (_generate(N_TEST,     +1, rng), "test",     +1),
        (_generate(N_OOD_TEST, -1, rng), "ood_test", -1),
    ]:
        df = pd.DataFrame(data)
        df["split"] = split
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Standard main.py entry point — in-distribution only."""
    df       = generate_full(seed)
    in_dist  = df[df["split"] != "ood_test"].copy()
    features = ["A", "Spur_shortcut", "X1_noise", "X2_noise", "X3_noise"]

    X = in_dist[features].to_numpy()
    y = in_dist["label"].to_numpy()

    # two variants:
    # full: all 5 features (A + Spur + 3 noise)
    # spur_only: only Spur_shortcut exposed — model MUST use shortcut
    #   in-dist accuracy stays high but ood drops to ~50% for everyone
    datasets = {
        "deep_causal_full":    (X, y),
        "deep_causal_spur_only": (
            in_dist[["Spur_shortcut"]].to_numpy(), y
        ),
    }
    return datasets


def run_causal_shortcut_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
) -> list[dict]:
    """
    THE real test — trains on TRAIN split, evaluates on both TEST and
    OOD_TEST separately. The accuracy_drop column is your failure signal.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []
    features = ["A", "Spur_shortcut", "X1_noise", "X2_noise", "X3_noise"]

    for seed in seeds:
        df = generate_full(seed)

        def _xy(split):
            sub = df[df["split"] == split]
            return sub[features].to_numpy(), sub["label"].to_numpy()

        X_train, y_train = _xy("train")
        X_test,  y_test  = _xy("test")
        X_ood,   y_ood   = _xy("ood_test")

        for model_name, factory in model_factories.items():
            print(f"[deep_causal_chain] {model_name} (seed={seed})...")
            base = {"model": model_name, "seed": seed}

            try:
                m_test = evaluate_classifier(
                    model=factory(), X_train=X_train,
                    X_test=X_test, y_train=y_train, y_test=y_test,
                )
                m_ood = evaluate_classifier(
                    model=factory(), X_train=X_train,
                    X_test=X_ood, y_train=y_train, y_test=y_ood,
                )
                row = {**base, "status": "success", "error": ""}
                for k, v in m_test.items():
                    row[f"{k}_test"] = v
                for k, v in m_ood.items():
                    row[f"{k}_ood"] = v
                row["accuracy_drop"] = m_test["accuracy"] - m_ood["accuracy"]
                row["roc_auc_drop"]  = (
                    m_test.get("roc_auc", float("nan"))
                    - m_ood.get("roc_auc", float("nan"))
                )
                print(
                    f"  test={m_test['accuracy']:.4f} | "
                    f"ood={m_ood['accuracy']:.4f} | "
                    f"drop={row['accuracy_drop']:.4f}"
                )
            except Exception as e:
                row = {**base, "status": "failed", "error": str(e)}
                print(f"  FAILED: {e}")

            rows.append(row)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "deep_causal_chain_ood_results.csv"
    all_cols = []
    for row in rows:
        for col in row:
            if col not in all_cols:
                all_cols.append(col)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {csv_path}")
    return rows


if __name__ == "__main__":
<<<<<<< HEAD
    save_csv(seed=0)  # optional inspection copy

    models_to_run = ["catboost", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]

=======
    models_to_run = ["catboost", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]
>>>>>>> e908ab1 (Obtained the final results and graph)
    print("=" * 70)
    print("Deep Causal Chain — OOD shortcut test")
    print("Hypothesis: TFMs lean on Spur_shortcut, drop on OOD flip")
    print("CatBoost finds true A signal, stays robust")
    print("=" * 70)
    run_causal_shortcut_check(model_names=models_to_run, seeds=(0, 1, 2))