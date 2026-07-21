# Kate's Synthetic Datasets — Feature Interaction, SCM/Causal, Robustness

Generated for the Machine Learning Lab AutoML project (models: **RealMLP, CatBoost**,
and evaluated against team datasets on **TabPFN v1/v2/v3, TabICLv2, LimiX**).
Fixed random seeds throughout — fully reproducible via `generate_datasets.py`.

Each dataset is a **single CSV**, with a `split` column so you can filter directly
in pandas (`df[df.split == "train"]`), and a `label` column as the binary target.
All datasets use `n_train = 8000`, `n_test = 2000` (S1/S2 also add `n_ood_test = 2000`).

---

## F1 — Feature Interaction (Medium)
**File:** `F1.csv` (10,000 rows × 3 features)
**Rule:** `label = 1 if A*B + C > T else 0`, threshold `T` calibrated so positives ≈ 50%.
**Features:** `A, B, C` ~ Uniform(1, 10)
**Label noise:** 2% random flips (avoids a perfectly separable task)
**Positive rate:** ~50.5%

Tests whether a model can recover a *pairwise multiplicative* interaction rather
than treating `A`, `B`, `C` as independently informative (a purely additive
model — e.g. shallow trees without interaction splits, or MLPs without enough
capacity/inductive bias — should underperform here).

## F2 — Feature Interaction (Hard)
**File:** `F2.csv` (10,000 rows × 4 features)
**Rule:** `label = 1 if A*B*C - D > T else 0`, threshold `T` calibrated to ~50% positives.
**Features:** `A, B, C, D` ~ Uniform(1, 10)
**Label noise:** 4%
**Positive rate:** ~50.6%

A **triple-order** interaction (harder than F1's pairwise one) minus a linear
term. Good for testing degradation as interaction order increases.

---

## S1 — SCM / Causal Reasoning (Medium)
**File:** `S1.csv` (12,000 rows × 3 features + label)
**True causal chain:** `A → B → label`
  - `A ~ N(0,1)`
  - `B = 1.5·A + N(0, 0.5)`
  - `label = 1 if 1.2·B + N(0, 0.3) > 0 else 0`
**Shortcut feature:** `Spur_shortcut`, correlated with `A` at **ρ = +0.9** in
`train`/`test`, but **ρ = −0.9** in `ood_test` (sign flipped).

This is the key design: a model that learned the **true causal path** (`A`, `B`)
will keep working on `ood_test`. A model that latched onto `Spur_shortcut` as a
correlational proxy for `A` will see its accuracy **invert/collapse** on
`ood_test`. Compare `test` vs `ood_test` accuracy per model — the gap is your
causal-reasoning failure signal.
**Positive rate:** ~50% in all three splits.

## S2 — SCM / Causal Reasoning (Hard)
**File:** `S2.csv` (12,000 rows × 5 features + label)
**True causal chain:** `A → B → C → label` (one node deeper than S1)
  - `A ~ N(0,1)`; `B = 1.3·A + N(0,0.4)`; `C = 1.3·B + N(0,0.4)`
  - `label = 1 if 1.1·C + N(0,0.3) > 0 else 0`
**Two shortcut features:** `Spur1_shortcut` (tied to `A`), `Spur2_shortcut`
(tied to `B`), both at **ρ = +0.85** in train/test and **ρ = −0.85** in
`ood_test`.
**Positive rate:** ~50% in all three splits.

Harder version of S1 — two independent shortcuts at different depths of the
chain, testing whether a model can correctly propagate the causal signal
through an extra hop instead of anchoring on either shortcut.

---

## R1 — Robustness (Medium)
**File:** `R1.csv` (10,000 rows × 24 features + label)
**Informative rule:** `label = 1 if (A+B) > 10.01 AND (C*D) > 18.66`
**Features:** `A, B, C, D` ~ Uniform(0,10) are informative; **20** columns
`noise_1 … noise_20` ~ N(0,1) are pure random noise, unrelated to the label.
**Label noise:** 2%
**Positive rate:** ~25% (conjunction of two independent ~50% conditions)

## R2 — Robustness (Hard)
**File:** `R2.csv` (10,000 rows × 105 features + label)
**Informative rule:** `label = 1 if ((A>5) XOR (B>5)) AND (C+D-E) > 5.00`
**Features:** `A, B, C, D, E` ~ Uniform(0,10) are informative; **100** columns
`noise_1 … noise_100` ~ N(0,1) are pure random noise.
**Label noise:** 4%
**Positive rate:** ~26%

R1 → R2 lets you plot **accuracy/AUC vs. number of irrelevant features**
(20 → 100) per model — a common failure mode for models that don't do implicit
feature selection (e.g. plain MLP-style baselines) versus tree-based models or
attention-based tabular foundation models that can down-weight noise columns.

---

## Suggested failure-case analyses
1. **F1 vs F2:** accuracy drop as interaction order goes pairwise → triple.
2. **S1 / S2, test vs ood_test:** accuracy gap = reliance on spurious shortcut
   vs. true causal features. Also inspect feature importance / attention
   weights on `Spur_shortcut` if the model exposes them.
3. **R1 vs R2:** accuracy/AUC drop as noise dimensionality goes 20 → 100,
   holding the true rule's difficulty roughly constant.
4. Across all six, compare **RealMLP vs CatBoost** (your assigned models)
   against the tabular foundation models your teammates are running
   (TabPFN v1/v2/v3, TabICLv2, LimiX) on the *same* CSVs for a fair
   cross-model comparison.

## Reproducibility
`generate_datasets.py` (included) regenerates every file exactly
(`MASTER_SEED = 42`, separate offset seeds per dataset). `dataset_metadata.json`
records the exact calibrated thresholds and observed class balances used above.

## Notes / things to double check with your supervisor
- Class balance for R1/R2 is ~25% positive by construction (AND of two ~50%
  conditions) rather than 50/50 — flag this if your team wants matched priors
  across all 6 datasets; happy to recalibrate to exactly 50/50 if preferred.
- Label noise rates (2% for Medium, 4% for Hard) are a modeling choice on my
  end to avoid perfectly separable toy tasks — let me know if your project
  spec requires noiseless ground truth instead.
