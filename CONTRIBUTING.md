# Contributing

This repository contains research code and recorded experimental results. Changes should preserve the distinction between source code, generated artifacts, and scientific claims.

## Before opening a change

1. Work on the branch corresponding to the experiment.
2. Do not overwrite committed raw results unless the experiment was intentionally rerun.
3. Record the command, seed, model versions, and hardware-relevant settings for new runs.
4. Keep generated logs, caches, checkpoint files, and local editor settings out of Git.
5. Update the branch README when a command, dataset definition, or result interpretation changes.

## Code style

- Prefer small functions with explicit inputs and outputs.
- Use comments to explain experimental choices, not ordinary Python syntax.
- Keep dataset generation deterministic when a seed is supplied.
- Fit preprocessing and feature selection on training data only.
- Write long-running results incrementally so interrupted runs remain auditable.

## Results

Every result table or figure should be traceable to:

- a generating script;
- an input result file;
- a model and dataset configuration;
- the number of seeds or splits;
- the metric definition.

Do not mix results produced by different script revisions in one summary without recording their provenance.

## Scientific language

Use “observed,” “suggests,” or “is consistent with” for empirical patterns. Reserve causal language for controlled ablations. A model-specific failure requires evidence that the same task remains learnable for at least one comparator.
