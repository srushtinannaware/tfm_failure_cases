# TFM Failure Cases — Poster Evidence Record

Last consolidated: 1 August 2026

## Recommended title

**When Newer Is Not Better: Accuracy and Computational Failure Modes of Tabular Foundation Models on Wide, Short Data**

## Research question

Where do tabular foundation models fail on extremely wide, short datasets, particularly with 100 training rows and hundreds to thousands of features?

## Experimental setup

- Models: TabPFN-2.5, TabPFN-3, TabICL v2, and XGBoost.
- Main synthetic benchmark: 100 training rows, 500 test rows, increasing feature count.
- Primary metrics: regression R² and classification accuracy.
- Runtime measurements separate model fitting from prediction.
- Targeted regression replication: 100, 200, and 500 features across seeds 0, 1, and 2.
- Arithmetic-chain checks vary irrelevant width, train/test distribution, and chain depth.

## Headline finding 1 — Predictive failure around 200 features

Mean regression R² across three seeds:

| Features | TabPFN-2.5 | TabPFN-3 | TabICL v2 | XGBoost |
|---:|---:|---:|---:|---:|
| 100 | 0.990 | 0.889 | 0.912 | 0.444 |
| **200** | **0.956** | **0.724** | **0.601** | **0.237** |
| 500 | 0.179 | 0.139 | 0.162 | 0.029 |

At 200 features, TabPFN-2.5 wins on every tested seed:

| Seed | TabPFN-2.5 | TabPFN-3 | TabICL v2 | XGBoost |
|---:|---:|---:|---:|---:|
| 0 | 0.956 | 0.663 | 0.538 | 0.203 |
| 1 | 0.970 | 0.795 | 0.670 | 0.247 |
| 2 | 0.943 | 0.713 | 0.593 | 0.262 |

**Supported conclusion:** The older TabPFN-2.5 is substantially more robust than TabPFN-3, TabICL v2, and XGBoost on this specific wide, short synthetic regression setting. The task is demonstrably learnable because TabPFN-2.5 obtains R² near 0.96 across all three seeds.

**Do not overclaim:** These experiments do not yet establish which architectural mechanism causes the gap. Feature handling or subsampling near this dimensional range is a hypothesis requiring a controlled ablation.

## Headline finding 2 — Computational failure at extreme width

At 5,000 features in the single-seed classification benchmark:

| Model | Accuracy | Prediction time | Relative to TabPFN-3 |
|---|---:|---:|---:|
| TabPFN-2.5 | 0.554 | 45.81 s | 2.5× |
| TabPFN-3 | 0.524 | 18.22 s | 1× |
| **TabICL v2** | **0.610** | **1,775.51 s (29.6 min)** | **97.4×** |

TabICL v2 has the best observed classification accuracy at 5,000 features, yet prediction takes almost 30 minutes. It is approximately 38.8× slower than TabPFN-2.5 and 97.4× slower than TabPFN-3.

**Supported conclusion:** A model may remain statistically competitive while becoming computationally impractical.

Regression shows the same TabICL timing problem (1,741.73 seconds), but every model has R² near zero at 5,000 features. Classification therefore provides the cleaner accuracy-versus-runtime example.

## Supporting experiments

### Experiment 1 — Irrelevant-width sweep

The arithmetic generator keeps 20 active features fixed while increasing total width. Classification generally deteriorates as irrelevant columns increase. Arithmetic regression remains near or below R² = 0 for all models, so it is not evidence of a model-specific regression failure.

### Experiment 2 — Out-of-distribution shift

TabPFN-3 exhibits localized OOD regression instability at lower widths, including R² = -1.491 at 200 features. The comparison models remain closer to zero, but none solves the arithmetic regression task strongly. This is evidence of possible instability, not a strong standalone predictive failure case.

### Experiment 3 — Hierarchical depth

At 500 features, increasing chain depth from 5 to 20 lowers classification accuracy for TabPFN-2.5, TabPFN-3, and XGBoost. Because the degradation is not unique to a TFM, it is best interpreted as a general complexity boundary. Arithmetic regression is weak and non-monotonic across models.

## Real-world biomedical validation — required next experiment

Use a human blood DNA-methylation dataset with chronological age as a continuous target.

Planned protocol:

- Use a fixed, reproducible subset of at most 200 people.
- Evaluate 100, 200, and 500 CpG features.
- Within every cross-validation fold, rank features by variance using the training fold only.
- Apply the training-derived feature selection unchanged to the held-out fold.
- Use identical folds and selected feature sets for every model.
- Compare TabPFN-2.5, TabPFN-3, TabICL v2, XGBoost, and preferably a regularized linear baseline.
- Report repeated cross-validated R², MAE, RMSE, and runtime with uncertainty.

Preregistered hypothesis: **TabPFN-2.5 retains an advantage over TabPFN-3 and TabICL v2 around 200 features on a real biomedical regression task.**

Interpretation rules:

- TabPFN-2.5 wins consistently: real-world corroboration of the synthetic failure.
- Models perform similarly: the synthetic ranking may be generator-specific.
- Every model performs poorly: the dataset does not support a model-specific failure claim.
- Another model wins: report this as a boundary on the synthetic result.

Do not describe the real-world result as replication until the dataset, protocol, and results are complete.

## Limitations to state explicitly

- The primary 5,000-feature study uses a single random seed.
- The targeted accuracy finding uses three seeds, which is stronger but still a limited sample of generated datasets.
- Holding training rows at 100 confounds absolute width with the feature-to-row ratio.
- The main accuracy result comes from synthetic data and may depend on the generator.
- Arithmetic-chain regression is weak across all models and does not establish a useful width-dependent comparative failure.
- OOD classification accuracy may be influenced by changes in class balance; balanced metrics should be inspected before making strong OOD claims.
- Runtime was measured on one hardware/software environment and should be interpreted as practical system-specific evidence rather than a universal benchmark.
- No architectural ablation has established the mechanism behind TabPFN-2.5's advantage.

## Poster hierarchy

1. Largest result panel: targeted regression R² at 100, 200, and 500 features.
2. Second-largest panel: 5,000-feature classification accuracy versus prediction time.
3. Real-world validation panel: cross-validated biomedical regression, once complete.
4. Compact methodology/support panel: arithmetic Experiments 1–3.
5. Conclusion and limitations panel: distinguish established findings from proposed mechanisms.

## Two-sentence poster conclusion

On a replicated wide, short synthetic regression task, TabPFN-2.5 substantially outperformed the newer TabPFN-3 and TabICL v2 around 200 features, showing that newer tabular foundation models are not uniformly more robust. At 5,000 features, TabICL v2 retained competitive classification accuracy but required almost 30 minutes for prediction, demonstrating that statistical performance can conceal computational impracticality.
