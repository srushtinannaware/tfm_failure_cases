# Failure-case study: TabPFN-3 and TabICL v2 vs. TabPFN-2.5 on wide, short data

**Scope note**: this is currently scoped to exactly three models —
TabPFN-2.5, TabPFN-3, and TabICL v2. TabPFN v2 and a Random Forest
baseline were part of an earlier draft and have been removed for now;
they're parked for a later pass, not part of this comparison.

**Hypothesis.** On very wide, very short tabular data (few rows, thousands
of columns — e.g. 100 rows × 5,000 columns), TabPFN-3 and TabICL v2
degrade sharply relative to TabPFN-2.5, and the reason is architectural,
not just "transformers don't scale to many features."

## Why this is an architecture story, not just a scaling limit

Each model's own documentation gives away a different design ceiling on
feature count:

| Model | Max rows | Max features | Source |
| :--- | ---: | ---: | :--- |
| **TabPFN-2.5** | 100,000 | 2,000 | [Prior Labs](https://docs.priorlabs.ai/models) |
| **TabPFN-3** | 1,000,000 [^1] | 200 [^1] | [Prior Labs](https://docs.priorlabs.ai/changelog/tabpfn-3) |
| **TabICL v2** | — | ~100 [^2] | [TabICL](https://pypi.org/project/tabicl/) |

[^1]: TabPFN-3 trades rows against features rather than having one fixed ceiling: 1M × 200, or 100K × 2K, or 1K × 20K, depending on configuration.
[^2]: TabICL v2's pretraining distribution covers 2–100 columns; it claims, but doesn't guarantee, generalization beyond that.

TabPFN-3 was optimized to scale to a million *rows* — its default
row/feature trade-off spends that budget on rows, not columns, unless you
explicitly configure it otherwise. TabICL v2's entire pretraining
distribution tops out at 100 columns; 5,000 is 50x outside anything it was
trained on. TabPFN-2.5, by contrast, was specifically built for width — a
"20x increase in data cells" over v2, validated up to 2,000 features.

There's also a concrete mechanism, not just a size limit. TabPFN's own
source (`tabpfn/constants.py`) defines `AUTO_FEATURE_SUBSAMPLING_TOP_K =
150` and a trigger threshold of 200 features
([PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)). Above that
threshold, TabPFN doesn't attend over all columns at once — each ensemble
member gets a random/balanced subset of features (round-robin in v3, so no
feature is *always* excluded, but no single estimator sees more than a
slice), and predictions are combined across the ensemble. If a dataset's
signal is concentrated in a handful of columns, this barely matters — any
subset probably catches one of them. But if signal is spread across
hundreds of columns (closer to real wide data like gene-expression
panels), each ensemble member is working with an incomplete, randomly
chosen picture, and accuracy should degrade as the informative-to-total
column ratio shrinks with scale. That's the mechanism this experiment is
built to expose.

TabICL v2 has a related but distinct issue: its complexity is O(n² + nm²)
in rows (n) and columns (m)
([TabICL PyPI page](https://pypi.org/project/tabicl/)), and its synthetic
pretraining prior was never sampled with more than 100 columns, so its
per-column attention patterns are being asked to extrapolate into a
completely unseen regime.

Related literature backs the general high-dimensional weakness: TabPFN v2
has been reported to fail to beat plain logistic regression on
high-dimensional data because its complexity scales with both rows and
features simultaneously (see "A Closer Look at TabPFN v2,"
[arXiv:2502.17361](https://arxiv.org/pdf/2502.17361)), and a Numerai forum
post independently benchmarking TabPFN-2.5 vs. TabICL v2 notes neither
holds up at true large-scale width
([Numerai forum](https://forum.numer.ai/t/foundation-models-on-numerai-data-tabpfn-v2-5-and-tabicl-v2/8269)).

## Method

`data_generation.py` builds synthetic 100-row datasets sweeping column
count (50 → 5,000, log-spaced) for both classification and regression. The
number of informative columns scales with total column count (10% of
columns, capped 8–300) rather than staying fixed — this matters: if
informative signal stayed concentrated in a fixed handful of columns,
per-estimator feature subsampling wouldn't matter much no matter how wide
the dataset got. Scaling informative-column count with total width is
what actually stresses the subsampling mechanism, and mirrors real wide
data (gene expression, wide sensor arrays) more closely than a toy dataset
with 5 "real" signal columns buried in 5,000 pure-noise ones.

`models.py` wraps three estimators behind one interface: TabPFN-2.5,
TabPFN-3 (via `TabPFNClassifier.create_default_for_version` with the
matching `tabpfn.constants.ModelVersion`), and TabICL v2 (the current
default `tabicl` checkpoint). Both TabPFN variants run with
`ignore_pretraining_limits=True` — the whole point of a failure-case study
is to push past each model's stated comfort zone and see what actually
happens, not to stay safely inside it.

`run_benchmark.py` sweeps task × column-count × seed × model, recording
accuracy/ROC-AUC (classification) or R²/RMSE (regression) plus fit and
predict wall-clock time, writing incrementally to `results/results.csv`.
`plot_results.py` turns that into the line graphs.

## Noise composition of the synthetic data

Informative-column count is `round(10% of total columns)`, clamped between
a floor of 8 and a ceiling of 300 (`_n_informative()` in
`data_generation.py`). The floor keeps small datasets from being
signal-starved; the ceiling means informative count stops growing past
3,000 columns even though total width keeps climbing — so the 5,000-column
dataset is proportionally noisier than the 1,000-column one, not just
larger.

| Columns | Informative | Redundant[^3] | Noise — classification | Noise — regression |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 8 | 8 | 34 (68%) | 42 (84%) |
| 100 | 10 | 10 | 80 (80%) | 90 (90%) |
| 500 | 50 | 50 | 400 (80%) | 450 (90%) |
| 1,000 | 100 | 100 | 800 (80%) | 900 (90%) |
| 5,000 | 300 (capped) | 300 | 4,400 (88%) | 4,700 (94%) |

[^3]: Redundant = linear combinations of informative columns, added by
`make_classification`'s `n_redundant` parameter; classification only.
`make_regression` has no equivalent, so regression's "noise" column is
simply everything past the informative count.

This is deliberate, not a confound to explain away: if informative signal
stayed fixed at a handful of columns regardless of width, TabPFN's
per-estimator feature subsampling and TabICL's column attention would
barely be stressed, because almost any random slice of columns would still
catch a real one. Scaling informative count with width keeps signal
genuinely spread out as the dataset grows, which is what actually exercises
the mechanism this study is testing — and mirrors real wide data (gene
expression panels, sensor arrays) more closely than a toy dataset with a
handful of "real" columns buried in thousands of noise ones.

**Open confound.** Because informative count and row count both interact
with total width in the same sweep, results here can't fully separate
"harder because there's more noise to sift through" from "harder because
100 rows can't estimate relationships across this many true dimensions,
regardless of noise." Resolving that, and checking whether the models are
actually attending to informative/redundant columns rather than latching
onto noise coincidentally, needs a feature-attribution pass (e.g. SHAP) —
deliberately deferred until after this first sweep.

## Results

Full single-seed sweep complete (seed=0, 100 train rows, 500 test rows,
7 column counts × 3 models × 2 tasks = 42 runs, all `status=ok`). Two
distinct failure modes showed up, not one:

**Accuracy failure (regression, 100–500 columns).** TabPFN-2.5 is clearly
the most robust: R²=0.956 at 200 columns vs. 0.663 (TabPFN-3) and 0.538
(TabICL v2). This is the strongest single piece of evidence for the
original hypothesis. By 1,000 columns all three collapse to near-zero R²
regardless of architecture — there's no signal left to preserve, so the
difference stops mattering. TabICL v2 is the only model to go negative
(R²=-0.088 at 2,000 columns), a qualitatively worse failure than "no
signal" — its predictions are actively anti-correlated with the target.

**Accuracy in classification is messier.** All three models track closely
through 200 columns (~0.92–0.98 accuracy) and degrade together up to
1,000 columns. Past that point they diverge: TabICL v2 ends up the *best*
performer at 5,000 columns (0.61), while TabPFN-3 — the newest model —
becomes the *worst* (0.524), a full reversal from being tied-best at low
column counts. TabPFN-2.5 sits in between (0.554). So the "newer
architecture degrades relatively worse" pattern holds for TabPFN-3 but not
for TabICL in this task.

**Computational failure (predict time, all tasks, high columns) — the
cleanest and most dramatic finding.** Through 200 columns, predict time is
close across all three models. Past that, TabICL v2's predict time
explodes non-linearly while both TabPFN variants stay comparatively flat:
at 5,000 columns, TabICL v2 takes ~1,776s (~30 min) to predict on 500 test
rows, vs. 45.8s (TabPFN-2.5) and 18.2s (TabPFN-3) — 40–100x longer. This
lines up with the architecture story above: TabPFN's per-estimator
feature subsampling structurally caps how many columns any single
ensemble member processes, bounding its cost; TabICL v2 has no equivalent
cap, so its O(n² + nm²) column-attention cost scales directly with total
feature count. TabICL v2's accuracy at this same extreme (0.61) was
actually competitive with the other two models — its failure mode isn't
wrong answers, it's becoming computationally impractical.

**Limitations.** All results above come from a single seed (`seed=0`), one
full pass through the sweep; seeds are 0-indexed, so no second or third
seed has been run yet. At 100 training rows, seed-to-seed variance is
real, and some of the finer-grained crossovers (e.g. exactly where TabICL
overtakes TabPFN-3 in classification) should be treated as illustrative
rather than proven until confirmed across multiple seeds (`--seeds 0 1 2`,
see `SETUP.md` §5). The regression robustness gap and the predict-time
blowup are large enough relative to expected seed noise to be trustworthy
as-is; the exact classification crossover point is not.

See `results/results.csv` for the raw numbers and `results/*.png` for the
plots (`classification_accuracy_vs_columns.png`,
`regression_r2_vs_columns.png`, `fit_time_vs_columns.png`,
`predict_time_vs_columns.png`, `timing_vs_columns.png`).

## Files

- `data_generation.py` — synthetic wide/short dataset generator
- `models.py` — unified model wrappers for the three in-scope models
- `run_benchmark.py` — sweep runner, writes `results/results.csv`
- `plot_results.py` — turns results into line graphs
- `SETUP.md` — how to run this somewhere with real Hugging Face access
- `results/` — `results.csv` (42-row single-seed sweep) plus five plots
- `failure_case_presentation.pptx` — slide deck summarizing this document
