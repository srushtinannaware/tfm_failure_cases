"""
Complex Logical Feature Interaction Dataset:
Implements two specific failure-case designs suggested by supervisor:

Design 1 — Repeated Feature XOR + Arithmetic:
    ((A > 5) XOR (B > 5)) AND ((A + C + D - E) > 5.0)
    The same feature A appears in BOTH the XOR clause and the arithmetic
    sum. This tests whether models can correctly integrate one feature
    playing two different roles simultaneously — a logical threshold role
    AND a continuous arithmetic role. TFMs trained on simple priors
    (one feature = one role) may fail to learn this dual-role pattern.

Design 2 — Mixed Multi-Operator Rules:
    Multiple logical operators (AND, OR, XOR, NOT) combined with
    arithmetic expressions in a single label rule. Instead of one
    clean operator per dataset, the rule chains several together:
        STAGE 1 (arithmetic): P = (A * B) + C > threshold_1
        STAGE 2 (logical OR): Q = (D > 6) OR (E < 2)
        STAGE 3 (logical NOT): R = NOT (F > 7)
        FINAL (AND all):      label = P AND Q AND R
    This creates a non-separable, multi-stage reasoning requirement
    that TFMs with smooth priors struggle to capture in one pass.

Why these specifically hurt TFMs:
    TabPFN / TabICL priors were trained on datasets generated from
    Bayesian Neural Networks and Structural Causal Models. These priors
    produce smooth, monotonic relationships. Discrete logical operators
    (especially XOR and NOT) create sharp, non-monotonic boundaries
    that are architecturally mismatched to the prior. CatBoost builds
    decision trees which naturally represent IF-THEN-ELSE logic and
    handles these rules trivially. RealMLP can approximate them with
    enough neurons, but may need tuning.

Variants per design:
    Design 1:
        xor_arithmetic_clean      — base rule, no noise
        xor_arithmetic_noisy      — 10 extra irrelevant features added
        xor_arithmetic_scaled     — features at mismatched scales
    Design 2:
        multi_operator_clean      — base rule, no noise
        multi_operator_noisy      — 10 extra irrelevant features added
        multi_operator_overlap    — features reused across multiple stages

Usage 1 (Standard main.py harness):
    python main.py --dataset complex_logical_interaction \\
        --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

Usage 2 (Standalone runner):
    python -m datasets.complex_logical_interaction
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TRAIN            = 1000
N_TEST             = 400
MASTER_SEED_OFFSET = 700   # unique stream — no overlap with other modules
LABEL_NOISE        = 0.02  # 2% random label flips, same as other kate files


# ---------------------------------------------------------------------------
# Shared helper: apply label noise
# ---------------------------------------------------------------------------
def _add_label_noise(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = y.copy()
    n_flip   = int(round(LABEL_NOISE * len(y)))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return y


# ---------------------------------------------------------------------------
# DESIGN 1 — Repeated Feature XOR + Arithmetic
#
# Rule: ((A > 5) XOR (B > 5)) AND ((A + C + D - E) > 5.0)
#
# Key point: feature A appears in BOTH clauses.
#   - In the XOR clause it acts as a THRESHOLD (is it above 5?)
#   - In the arithmetic clause it acts as a CONTINUOUS value (add it to sum)
# This dual-role usage is the failure trigger for TFMs.
# ---------------------------------------------------------------------------
def _xor_arithmetic_rule(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
) -> np.ndarray:
    """
    ((A > 5) XOR (B > 5)) AND ((A + C + D - E) > 5.0)

    XOR truth table reminder:
        A>5=True,  B>5=True  → XOR=False → label=0
        A>5=True,  B>5=False → XOR=True  → depends on arithmetic
        A>5=False, B>5=True  → XOR=True  → depends on arithmetic
        A>5=False, B>5=False → XOR=False → label=0
    """
    xor_clause        = np.logical_xor(A > 5, B > 5)
    arithmetic_clause = (A + C + D - E) > 5.0
    return (xor_clause & arithmetic_clause).astype(int)


def _generate_design1_clean(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Base variant — 5 core features, no noise."""
    A = rng.uniform(0, 10, n)   # appears in BOTH clauses (the dual-role feature)
    B = rng.uniform(0, 10, n)   # threshold-only role
    C = rng.uniform(0, 10, n)   # arithmetic-only role
    D = rng.uniform(0, 10, n)   # arithmetic-only role
    E = rng.uniform(0, 10, n)   # arithmetic-only role (subtracted)

    y = _xor_arithmetic_rule(A, B, C, D, E)
    y = _add_label_noise(y, rng)

    X = np.stack([A, B, C, D, E], axis=1)
    return X, y


def _generate_design1_noisy(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy variant — same rule + 10 pure noise features appended."""
    X_core, y = _generate_design1_clean(n, rng)
    noise      = rng.normal(0, 1, (n, 10))
    return np.hstack([X_core, noise]), y


def _generate_design1_scaled(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale-mismatch variant — same rule but features presented at
    wildly different magnitudes. A (the dual-role feature) stays at
    the original [0,10] range. B is blown up to [0, 1e4]. C is
    compressed to [0, 1e-3]. D and E stay near original scale.

    The TRUE rule still only cares about standardized thresholds,
    but the raw scale difference should confuse TFM embeddings.
    """
    A = rng.uniform(0,    10,   n)   # original scale — dual-role
    B = rng.uniform(0,    10,   n) * 1e3   # blown up
    C = rng.uniform(0,    10,   n) * 1e-3  # compressed
    D = rng.uniform(0,    10,   n)
    E = rng.uniform(0,    10,   n)

    # label is computed on ORIGINAL standardized values
    # (same rule, scale applied only to what the model sees)
    A_std = A                        # already [0,10]
    B_std = B / 1e3                  # bring back to [0,10] for rule
    C_std = C / 1e-3                 # bring back to [0,10] for rule

    y = _xor_arithmetic_rule(A_std, B_std, C_std, D, E)
    y = _add_label_noise(y, rng)

    X = np.stack([A, B, C, D, E], axis=1)   # model sees raw scales
    return X, y


# ---------------------------------------------------------------------------
# DESIGN 2 — Mixed Multi-Operator Rules
#
# Rule (3 stages, all must be true):
#   P = (A * B + C)  > threshold_P          [arithmetic + threshold]
#   Q = (D > 6.0) OR (E < 2.0)             [logical OR]
#   R = NOT (F > 7.0)                       [logical NOT]
#   label = P AND Q AND R
#
# Each stage uses a different operator type. The final label requires
# ALL three stages to be satisfied simultaneously.
# ---------------------------------------------------------------------------
def _multi_operator_rule(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    F: np.ndarray,
    threshold_P: float = 10.0,
) -> np.ndarray:
    """
    STAGE 1 — Arithmetic + threshold:
        P = (A * B + C) > threshold_P

    STAGE 2 — Logical OR:
        Q = (D > 6.0) OR (E < 2.0)

    STAGE 3 — Logical NOT:
        R = NOT (F > 7.0)

    FINAL — AND all three:
        label = P AND Q AND R
    """
    P = (A * B + C) > threshold_P
    Q = (D > 6.0) | (E < 2.0)
    R = ~(F > 7.0)
    return (P & Q & R).astype(int)


def _generate_design2_clean(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Base variant — 6 core features, no noise, no overlap."""
    A = rng.uniform(0, 10, n)
    B = rng.uniform(0, 10, n)
    C = rng.uniform(0, 10, n)
    D = rng.uniform(0, 10, n)
    E = rng.uniform(0, 10, n)
    F = rng.uniform(0, 10, n)

    y = _multi_operator_rule(A, B, C, D, E, F)
    y = _add_label_noise(y, rng)

    X = np.stack([A, B, C, D, E, F], axis=1)
    return X, y


def _generate_design2_noisy(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy variant — same rule + 10 pure noise features appended."""
    X_core, y = _generate_design2_clean(n, rng)
    noise      = rng.normal(0, 1, (n, 10))
    return np.hstack([X_core, noise]), y


def _generate_design2_overlap(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlap variant — same features are REUSED across multiple stages.
    Feature A appears in BOTH Stage 1 (arithmetic) AND Stage 2 (OR threshold).
    Feature D appears in BOTH Stage 2 (OR threshold) AND Stage 3 (NOT threshold).

    Updated rule:
        P = (A * B + C) > 10.0           [A used here]
        Q = (A > 6.0) OR (E < 2.0)       [A reused here — different role]
        R = NOT (D > 7.0)                 [D used here]
        label = P AND Q AND R

    This maximises the dual-role effect from Design 1 across ALL stages.
    """
    A = rng.uniform(0, 10, n)   # dual role: arithmetic (P) + OR threshold (Q)
    B = rng.uniform(0, 10, n)
    C = rng.uniform(0, 10, n)
    D = rng.uniform(0, 10, n)   # appears in stage 2 and stage 3
    E = rng.uniform(0, 10, n)

    P = (A * B + C) > 10.0          # arithmetic
    Q = (A > 6.0) | (E < 2.0)      # A reused as threshold
    R = ~(D > 7.0)                   # D as NOT threshold

    y = (P & Q & R).astype(int)
    y = _add_label_noise(y, rng)

    X = np.stack([A, B, C, D, E], axis=1)
    return X, y


# ---------------------------------------------------------------------------
# main.py entry point
# ---------------------------------------------------------------------------
def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns all 6 variants (3 per design).
    Named with clear prefixes so graphs are self-explanatory.
    """
    rng      = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total  = N_TRAIN + N_TEST
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── Design 1: XOR + Arithmetic with repeated feature A ───────────────
    datasets["xor_arith_clean"]   = _generate_design1_clean(n_total, rng)
    datasets["xor_arith_noisy"]   = _generate_design1_noisy(n_total, rng)
    datasets["xor_arith_scaled"]  = _generate_design1_scaled(n_total, rng)

    # ── Design 2: Mixed multi-operator (AND + OR + NOT + arithmetic) ──────
    datasets["multi_op_clean"]    = _generate_design2_clean(n_total, rng)
    datasets["multi_op_noisy"]    = _generate_design2_noisy(n_total, rng)
    datasets["multi_op_overlap"]  = _generate_design2_overlap(n_total, rng)

    return datasets


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def run_complex_logical_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
) -> list[dict]:
    """
    Runs all 6 variants across all models and seeds.
    Saves results/complex_logical_interaction_results.csv
    """
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

            # log class balance so we can check if rule is degenerate
            pos_rate = float(y_train.mean())

            for model_name, factory in model_factories.items():
                print(
                    f"[{variant_name}] {model_name} "
                    f"(seed={seed}, pos_rate={pos_rate:.2f})..."
                )

                base_row = {
                    "dataset_variant" : variant_name,
                    "model"           : model_name,
                    "seed"            : seed,
                    "n_train"         : N_TRAIN,
                    "n_test"          : N_TEST,
                    "n_features"      : int(X.shape[1]),
                    "pos_rate_train"  : round(pos_rate, 4),
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
                        f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                        f"Time={metrics['total_seconds']:.2f}s"
                    )

                except Exception as error:
                    row = {**base_row,
                           "status": "failed", "error": str(error)}
                    print(f"  FAILED: {error}")

                rows.append(row)

    # ── save ──────────────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = results_dir / "complex_logical_interaction_results.csv"

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
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Confirm class balance is reasonable before running full benchmark
    print("Sanity check — class balance per variant (seed=0):")
    rng_check = np.random.default_rng(0 + MASTER_SEED_OFFSET)
    n_check   = N_TRAIN + N_TEST
    checks = {
        "xor_arith_clean"  : _generate_design1_clean(n_check, rng_check),
        "xor_arith_noisy"  : _generate_design1_noisy(n_check, rng_check),
        "xor_arith_scaled" : _generate_design1_scaled(n_check, rng_check),
        "multi_op_clean"   : _generate_design2_clean(n_check, rng_check),
        "multi_op_noisy"   : _generate_design2_noisy(n_check, rng_check),
        "multi_op_overlap" : _generate_design2_overlap(n_check, rng_check),
    }
    for name, (X, y) in checks.items():
        print(f"  {name:<25} shape={X.shape}  "
              f"pos={y.mean():.2f}  neg={1-y.mean():.2f}")

    print()
    print("Running full benchmark...")
    print("=" * 70)

    models_to_run = [
        "catboost",
        "realmlp",
        "tabpfn_v2",
        "tabpfn_v3",
        "tabicl_v2",
    ]
    run_complex_logical_check(model_names=models_to_run)