"""
Sparse Hierarchical Logic Dataset:
Tests reasoning DEPTH under fixed sparsity, decoupled from dimensionality.

A small, fixed pool of K sparse features is reused across a chain of D
composed binary tests (b_d = X[:, d % K] > 0), combined by a running
XOR (with an AND every 3rd step for variety) to produce the label.
Total feature count and sparsity (K) are held CONSTANT across variants;
only the chain length D — the number of conditions that must be
composed correctly — increases. This isolates "how many sequential
logical steps can the model chain" from "how many raw dimensions are
there", which the existing attention-dilution dataset already covers.

Why not plain XOR/parity: a recent probing paper on TabPFN-v2
("What exactly has TabPFN learned to do?", arXiv 2502.08978) found it
approximately learns the parity function — so a single XOR of a few
features is NOT a safe assumption for a clean TabPFN failure. This is
why the dataset sweeps D explicitly rather than shipping one XOR
variant and hoping it fails; you're looking for the depth at which the
degradation actually starts, which is an empirical question this
script lets you answer per model.

TODO(kate): the AND-every-3rd-step choice below is arbitrary — if you
want a pure parity chain instead (to compare directly against the
2502.08978 result), set `and_every=0`.

Variants:
  - hier_depth_2 / 4 / 6 / 8 / 10 / 12: same K=6 feature pool,
    increasing composed-condition chain length D.
  - hier_depth_8_wide: same D=8, but pool widened to 30 features
    (only 8 of them used per row's chain, picked from a fixed
    per-variant subset) — isolates "chain length" from "you could
    brute-force search a small pool" as an alternate explanation for
    any tree-vs-ICL gap you see.

Usage:
    python main.py --dataset sparse_hierarchical_logic --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
    python -m datasets.sparse_hierarchical_logic
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

N_TRAIN = 1000
N_TEST = 200
MASTER_SEED_OFFSET = 700
LABEL_FLIP_PROB = 0.02


def _make_hierarchical(
    n: int,
    depth: int,
    pool_size: int,
    rng,
    and_every: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0, 1, size=(n, pool_size))
    K = min(pool_size, max(depth, 6))  # feature pool actually used in the chain
    feature_order = rng.permutation(pool_size)[:K]

    bits = np.zeros((n, depth), dtype=bool)
    for d in range(depth):
        col = feature_order[d % K]
        bits[:, d] = X[:, col] > 0

    label = bits[:, 0].copy()
    for d in range(1, depth):
        if and_every and d % and_every == 0:
            label = label & bits[:, d]
        else:
            label = label ^ bits[:, d]

    y = label.astype(int)
    flips = rng.random(n) < LABEL_FLIP_PROB
    y = np.where(flips, 1 - y, y)
    return X, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets = {}

    # --- original sweep: kept for continuity, but note pool_size=6 means
    # depth > 6 reuses columns (bits[:, d] == bits[:, d+6] exactly), which
    # confounds "depth" with "repeated/cancelling composition" -- this is
    # almost certainly why every model (incl. CatBoost) dips at depth_6 and
    # partially recovers at depth_8/10. Do not read this as a depth effect.
    for depth in (2, 4, 6, 8, 10, 12):
        X, y = _make_hierarchical(n_total, depth, pool_size=6, rng=rng)
        datasets[f"hier_depth_{depth}"] = (X, y)

    X, y = _make_hierarchical(n_total, depth=8, pool_size=30, rng=rng)
    datasets["hier_depth_8_wide"] = (X, y)

    # --- PRIMARY clean depth sweep: pool_size == depth (fresh column every
    # step, no reuse/cancellation confound) AND and_every=0 (pure parity,
    # so labels are guaranteed exactly 50/50 at every depth -- unlike the
    # AND-mixed chain, whose label balance swings with d % and_every in a
    # way that doesn't track depth and would confound accuracy comparisons
    # across variants). This is also a direct comparison point against the
    # arXiv 2502.08978 finding that TabPFN-v2 approximates parity well: if
    # parity holds up at depth while the AND-mixed version below degrades,
    # that pins the failure on "long mixed boolean composition" specifically,
    # not on chain length or parity per se.
    for depth in (4, 8, 12, 16, 20, 24, 32):
        X, y = _make_hierarchical(n_total, depth, pool_size=depth, rng=rng, and_every=0)
        datasets[f"hier_purestep_parity_depth_{depth}"] = (X, y)

    # --- SECONDARY, explicitly imbalanced: same fresh-column setup but with
    # ANDs mixed in (and_every=3). Label balance is NOT 50/50 here and shifts
    # with depth (verified empirically: e.g. depth=4 -> ~25% positive,
    # depth=8 -> ~52%) -- do not compare raw accuracy against the parity
    # sweep above or across these depths without also checking/reporting
    # each variant's label rate, and prefer balanced-accuracy or AUC over
    # raw accuracy when analyzing this group.
    for depth in (4, 8, 12, 16, 20, 24, 32):
        X, y = _make_hierarchical(n_total, depth, pool_size=depth, rng=rng, and_every=3)
        datasets[f"hier_purestep_mixed_depth_{depth}"] = (X, y)

    return datasets


def run_hierarchical_logic_check(model_names: list[str], seeds: list[int] = (0, 1, 2)) -> list[dict]:
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
    csv_path = results_dir / "sparse_hierarchical_logic_results.csv"

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
    print("Running Sparse Hierarchical Logic Benchmark")
    print("=" * 70)
    run_hierarchical_logic_check(model_names=models_to_run)
