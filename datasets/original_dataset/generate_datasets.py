"""
Synthetic dataset generator for Kate's assigned reasoning families:
  - Feature Interaction  (F1, F2)
  - SCM / Causal Reasoning (S1, S2)
  - Robustness           (R1, R2)

Designed to be used with tabular foundation models (TabPFN v1/v2/v3, TabICLv2,
LimiX) and baselines (CatBoost, RealMLP) to surface failure modes tied to each
reasoning family.

Design notes
------------
F1 / F2   : pure feature-interaction rules (pairwise / higher-order products).
            Thresholds are CALIBRATED on a large held-out calibration sample so
            the positive class rate lands close to 50% (avoids trivial
            majority-class shortcuts). Small label noise is injected to avoid
            a perfectly separable / degenerate task.

S1 / S2   : genuine causal chains (A->B->Label, A->B->C->Label) PLUS a spurious
            "shortcut" feature that is strongly correlated with a true causal
            node during train/iid-test, but has that correlation REVERSED in a
            separate ood_test split. A model that has learned the true causal
            mechanism (A, B, [C]) keeps working on ood_test; a model that
            leaned on the spurious shortcut feature degrades sharply. This is
            the standard way to probe causal vs. correlational shortcut
            learning in an SCM benchmark.

R1 / R2   : a moderately complex ground-truth rule over a small set of
            informative features, with 20 (R1) or 100 (R2) purely random,
            label-irrelevant noise columns appended. Tests resilience to
            irrelevant/high-dimensional noise.

Every dataset is written as ONE self-contained CSV with a `split` column
(`train` / `test` / `ood_test` where applicable) so each dataset lives in its
own file as requested, and each file is directly usable end-to-end.
"""

import numpy as np
import pandas as pd
import os
import json

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

MASTER_SEED = 42
N_TRAIN = 1000
N_TEST = 200
N_OOD_TEST = 200
LABEL_NOISE_MEDIUM = 0.02
LABEL_NOISE_HARD = 0.04

summary = {}  # collects metadata / class balance for the README


def flip_labels(y, noise_rate, rng):
    y = y.copy()
    n = len(y)
    n_flip = int(round(noise_rate * n))
    idx = rng.choice(n, size=n_flip, replace=False)
    y[idx] = 1 - y[idx]
    return y


def calibrate_threshold(score_fn, rng, target_rate=0.5, n_calib=300_000):
    """Draw a large calibration sample and return the score quantile that
    yields ~target_rate positives, for balanced-ish binary labels."""
    scores = score_fn(n_calib, rng)
    return np.quantile(scores, 1 - target_rate)


# ---------------------------------------------------------------------------
# F1 — Feature Interaction (Medium): pairwise interaction  A*B + C > T
# ---------------------------------------------------------------------------
def make_F1():
    rng = np.random.default_rng(MASTER_SEED + 1)

    def score_fn(n, r):
        A = r.uniform(1, 10, n)
        B = r.uniform(1, 10, n)
        C = r.uniform(1, 10, n)
        return A * B + C

    T = calibrate_threshold(score_fn, rng)

    n_total = N_TRAIN + N_TEST
    A = rng.uniform(1, 10, n_total)
    B = rng.uniform(1, 10, n_total)
    C = rng.uniform(1, 10, n_total)
    raw_score = A * B + C
    y = (raw_score > T).astype(int)
    y = flip_labels(y, LABEL_NOISE_MEDIUM, rng)

    split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST)
    df = pd.DataFrame({"A": A, "B": B, "C": C, "label": y, "split": split})

    summary["F1"] = dict(
        family="Feature Interaction", difficulty="Medium",
        rule="label = 1[A*B + C > T]  (T calibrated for ~50% positive rate)",
        threshold_T=round(float(T), 3),
        features=["A", "B", "C"], target="label",
        n_train=N_TRAIN, n_test=N_TEST,
        label_noise=LABEL_NOISE_MEDIUM,
        positive_rate=round(float(y.mean()), 4),
    )
    return df


# ---------------------------------------------------------------------------
# F2 — Feature Interaction (Hard): higher-order interaction  A*B*C - D > T
# ---------------------------------------------------------------------------
def make_F2():
    rng = np.random.default_rng(MASTER_SEED + 2)

    def score_fn(n, r):
        A = r.uniform(1, 10, n)
        B = r.uniform(1, 10, n)
        C = r.uniform(1, 10, n)
        D = r.uniform(1, 10, n)
        return A * B * C - D

    T = calibrate_threshold(score_fn, rng)

    n_total = N_TRAIN + N_TEST
    A = rng.uniform(1, 10, n_total)
    B = rng.uniform(1, 10, n_total)
    C = rng.uniform(1, 10, n_total)
    D = rng.uniform(1, 10, n_total)
    raw_score = A * B * C - D
    y = (raw_score > T).astype(int)
    y = flip_labels(y, LABEL_NOISE_HARD, rng)

    split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST)
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "label": y, "split": split})

    summary["F2"] = dict(
        family="Feature Interaction", difficulty="Hard",
        rule="label = 1[A*B*C - D > T]  (T calibrated for ~50% positive rate)",
        threshold_T=round(float(T), 3),
        features=["A", "B", "C", "D"], target="label",
        n_train=N_TRAIN, n_test=N_TEST,
        label_noise=LABEL_NOISE_HARD,
        positive_rate=round(float(y.mean()), 4),
    )
    return df


# ---------------------------------------------------------------------------
# S1 — SCM / Causal Reasoning (Medium): chain A -> B -> Label, + 1 spurious
#      shortcut feature whose correlation with A flips sign in ood_test.
# ---------------------------------------------------------------------------
def make_S1():
    rng = np.random.default_rng(MASTER_SEED + 3)
    rho = 0.9

    def gen(n, rho_sign, r):
        A = r.normal(0, 1, n)
        Spur = rho_sign * rho * A + np.sqrt(1 - rho ** 2) * r.normal(0, 1, n)
        B = 1.5 * A + r.normal(0, 0.5, n)
        logit_noise = r.normal(0, 0.3, n)
        y = (1.2 * B + logit_noise > 0).astype(int)
        return A, Spur, B, y

    A_tr, Sp_tr, B_tr, y_tr = gen(N_TRAIN, +1, rng)
    A_te, Sp_te, B_te, y_te = gen(N_TEST, +1, rng)
    A_ood, Sp_ood, B_ood, y_ood = gen(N_OOD_TEST, -1, rng)  # spurious corr. flipped

    A = np.concatenate([A_tr, A_te, A_ood])
    Spur = np.concatenate([Sp_tr, Sp_te, Sp_ood])
    B = np.concatenate([B_tr, B_te, B_ood])
    y = np.concatenate([y_tr, y_te, y_ood])
    split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST + ["ood_test"] * N_OOD_TEST)

    df = pd.DataFrame({"A": A, "Spur_shortcut": Spur, "B": B, "label": y, "split": split})

    summary["S1"] = dict(
        family="SCM / Causal Reasoning", difficulty="Medium",
        rule="Causal chain A -> B -> label (B=1.5A+noise; label=1[1.2B+noise>0]). "
             "Spur_shortcut correlates with A at rho=+0.9 in train/test, "
             "rho=-0.9 in ood_test — a model relying on the shortcut instead "
             "of the true causal path (A, B) should degrade on ood_test.",
        features=["A", "Spur_shortcut", "B"], target="label",
        n_train=N_TRAIN, n_test=N_TEST, n_ood_test=N_OOD_TEST,
        positive_rate_train=round(float(y_tr.mean()), 4),
        positive_rate_test=round(float(y_te.mean()), 4),
        positive_rate_ood_test=round(float(y_ood.mean()), 4),
    )
    return df


# ---------------------------------------------------------------------------
# S2 — SCM / Causal Reasoning (Hard): deeper chain A -> B -> C -> Label,
#      + 2 spurious shortcut features (at A and at B), both sign-flipped in
#      ood_test.
# ---------------------------------------------------------------------------
def make_S2():
    rng = np.random.default_rng(MASTER_SEED + 4)
    rho1, rho2 = 0.85, 0.85

    def gen(n, sign, r):
        A = r.normal(0, 1, n)
        Spur1 = sign * rho1 * A + np.sqrt(1 - rho1 ** 2) * r.normal(0, 1, n)
        B = 1.3 * A + r.normal(0, 0.4, n)
        Spur2 = sign * rho2 * B + np.sqrt(1 - rho2 ** 2) * r.normal(0, 1, n)
        C = 1.3 * B + r.normal(0, 0.4, n)
        logit_noise = r.normal(0, 0.3, n)
        y = (1.1 * C + logit_noise > 0).astype(int)
        return A, Spur1, B, Spur2, C, y

    A_tr, S1_tr, B_tr, S2_tr, C_tr, y_tr = gen(N_TRAIN, +1, rng)
    A_te, S1_te, B_te, S2_te, C_te, y_te = gen(N_TEST, +1, rng)
    A_od, S1_od, B_od, S2_od, C_od, y_od = gen(N_OOD_TEST, -1, rng)

    A = np.concatenate([A_tr, A_te, A_od])
    Spur1 = np.concatenate([S1_tr, S1_te, S1_od])
    B = np.concatenate([B_tr, B_te, B_od])
    Spur2 = np.concatenate([S2_tr, S2_te, S2_od])
    C = np.concatenate([C_tr, C_te, C_od])
    y = np.concatenate([y_tr, y_te, y_od])
    split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST + ["ood_test"] * N_OOD_TEST)

    df = pd.DataFrame({
        "A": A, "Spur1_shortcut": Spur1, "B": B, "Spur2_shortcut": Spur2,
        "C": C, "label": y, "split": split
    })

    summary["S2"] = dict(
        family="SCM / Causal Reasoning", difficulty="Hard",
        rule="Deeper causal chain A -> B -> C -> label "
             "(B=1.3A+noise; C=1.3B+noise; label=1[1.1C+noise>0]). "
             "Two shortcut features (Spur1 tied to A, Spur2 tied to B) at "
             "rho=+0.85 in train/test, rho=-0.85 in ood_test.",
        features=["A", "Spur1_shortcut", "B", "Spur2_shortcut", "C"], target="label",
        n_train=N_TRAIN, n_test=N_TEST, n_ood_test=N_OOD_TEST,
        positive_rate_train=round(float(y_tr.mean()), 4),
        positive_rate_test=round(float(y_te.mean()), 4),
        positive_rate_ood_test=round(float(y_od.mean()), 4),
    )
    return df


# ---------------------------------------------------------------------------
# R1 — Robustness (Medium): informative rule over 4 features + 20 noise cols
# ---------------------------------------------------------------------------
def make_R1():
    rng = np.random.default_rng(MASTER_SEED + 5)
    N_NOISE = 20

    def score_parts(n, r):
        A = r.uniform(0, 10, n)
        B = r.uniform(0, 10, n)
        C = r.uniform(0, 10, n)
        D = r.uniform(0, 10, n)
        return A, B, C, D

    # calibrate the two sub-thresholds (sum-threshold, product-threshold)
    def calib_score(n, r):
        A, B, C, D = score_parts(n, r)
        return (A + B), (C * D)

    A_c, B_c, C_c, D_c = score_parts(300_000, rng)
    sum_thresh = np.quantile(A_c + B_c, 0.5)
    prod_thresh = np.quantile(C_c * D_c, 0.5)

    n_total = N_TRAIN + N_TEST
    A, B, C, D = score_parts(n_total, rng)
    y = (((A + B) > sum_thresh) & ((C * D) > prod_thresh)).astype(int)
    y = flip_labels(y, LABEL_NOISE_MEDIUM, rng)

    noise = rng.normal(0, 1, size=(n_total, N_NOISE))
    noise_cols = {f"noise_{i+1}": noise[:, i] for i in range(N_NOISE)}

    split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST)
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, **noise_cols,
                        "label": y, "split": split})

    summary["R1"] = dict(
        family="Robustness", difficulty="Medium",
        rule=f"label = 1[(A+B) > {sum_thresh:.2f}  AND  (C*D) > {prod_thresh:.2f}], "
             f"plus {N_NOISE} pure Gaussian noise columns (noise_1..noise_{N_NOISE}) "
             "with zero relationship to the label.",
        informative_features=["A", "B", "C", "D"],
        n_noise_features=N_NOISE, target="label",
        n_train=N_TRAIN, n_test=N_TEST,
        label_noise=LABEL_NOISE_MEDIUM,
        positive_rate=round(float(y.mean()), 4),
    )
    return df


# ---------------------------------------------------------------------------
# R2 — Robustness (Hard): XOR + arithmetic rule over 5 features + 100 noise
# ---------------------------------------------------------------------------
def make_R2():
    rng = np.random.default_rng(MASTER_SEED + 6)
    N_NOISE = 100

    def score_parts(n, r):
        A = r.uniform(0, 10, n)
        B = r.uniform(0, 10, n)
        C = r.uniform(0, 10, n)
        D = r.uniform(0, 10, n)
        E = r.uniform(0, 10, n)
        return A, B, C, D, E

    def calib_score(n, r):
        A, B, C, D, E = score_parts(n, r)
        return (C + D - E)

    A_c, B_c, C_c, D_c, E_c = score_parts(300_000, rng)
    arith_thresh = np.quantile(C_c + D_c - E_c, 0.5)

    n_total = N_TRAIN + N_TEST
    A, B, C, D, E = score_parts(n_total, rng)
    xor_part = ((A > 5).astype(int) != (B > 5).astype(int))  # XOR
    arith_part = (C + D - E) > arith_thresh
    y = (xor_part & arith_part).astype(int)
    y = flip_labels(y, LABEL_NOISE_HARD, rng)

    noise = rng.normal(0, 1, size=(n_total, N_NOISE))
    noise_cols = {f"noise_{i+1}": noise[:, i] for i in range(N_NOISE)}

    split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST)
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E, **noise_cols,
                        "label": y, "split": split})

    summary["R2"] = dict(
        family="Robustness", difficulty="Hard",
        rule=f"label = 1[ ((A>5) XOR (B>5))  AND  (C+D-E) > {arith_thresh:.2f} ], "
             f"plus {N_NOISE} pure Gaussian noise columns (noise_1..noise_{N_NOISE}) "
             "with zero relationship to the label.",
        informative_features=["A", "B", "C", "D", "E"],
        n_noise_features=N_NOISE, target="label",
        n_train=N_TRAIN, n_test=N_TEST,
        label_noise=LABEL_NOISE_HARD,
        positive_rate=round(float(y.mean()), 4),
    )
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generators = {
        "F1": make_F1,
        "F2": make_F2,
        "S1": make_S1,
        "S2": make_S2,
        "R1": make_R1,
        "R2": make_R2,
    }
    for ds_id, fn in generators.items():
        df = fn()
        out_path = os.path.join(OUT_DIR, f"{ds_id}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  shape={df.shape}")

    with open(os.path.join(OUT_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote dataset_metadata.json")