# Stress-testing tabular foundation models

This repository contains controlled synthetic benchmarks for studying failure modes of tabular foundation models (TFMs). The experiments vary one source of difficulty at a time: feature dimensionality, irrelevant-feature dilution, logical interactions, and examples available per categorical key.

The project was developed for the University of Freiburg AutoML course by Kate Pereira, Sudhanwa Bandi, and Srushti Nannaware, supervised by Alexander Pfefferle and Omar Swelam.

## Research question

Can controlled synthetic experiments uncover reproducible conditions in which the ranking of tabular foundation models changes?

The purpose is not to identify one universally best model. Instead, each benchmark asks whether a task remains learnable while a particular model or model family degrades.

## Main findings

### Wide, short regression

With 100 training rows and 200 features, TabPFN-2.5 consistently outperformed TabPFN-3, TabICL v2, and XGBoost across three generated datasets.

| Model | Mean test R² | Standard deviation |
|---|---:|---:|
| TabPFN-2.5 | 0.956 | 0.013 |
| TabPFN-3 | 0.724 | 0.067 |
| TabICL v2 | 0.601 | 0.066 |
| XGBoost | 0.237 | 0.031 |

This result is specific to the tested synthetic generator. It does not establish that TabPFN-2.5 is universally better on wide data.

### Multi-condition XOR under irrelevant features

The logical benchmark tests an XOR target whose signal is diluted by irrelevant variables. Several TFMs approach chance-level balanced accuracy as irrelevant features are added, while CatBoost remains stronger in the reported four-noise-feature condition.

### Random context routing

### Random context routing

The routing benchmark tests categorical lookup: each row's label is generated as `y = lookup[key]`, with no relationship between nearby key values and their labels, so a model can only score above chance by having seen that specific key a sufficient number of times during training. Ten pure-noise features and a 2% label-flip rate were added to the base task.

Difficulty was controlled through shots per key (training examples per unique key), computed as `N_train / n_keys`. `N_train` was fixed at 3,000 and `n_keys` was swept from 150 to 950 in steps of 150, moving from roughly 20 shots per key down to roughly 2.6. Evaluation used ROC-AUC across the full sweep and F1-macro at the sparsest setting, across two seeds.

At the sparsest setting tested (2.6 shots per key), F1-macro scores were: TabICL v2 0.83, CatBoost 0.54, TabPFN-3 0.46, TabPFN v2 0.41. Across the sweep, TabICL v2's ROC-AUC stayed above roughly 0.9 for most of the tested range, while ROC-AUC for the other three models declined as shots per key decreased.

Consistent with the evaluation principles above, this indicates a comparative advantage for TabICL v2 under sparse-key conditions rather than an established architectural cause — no attention or representation analysis was performed to confirm a specific mechanism.

## Repository branches

The study was developed as three independent failure-case investigations plus shared model wrappers. The main branch is the project landing page; executable experiment code lives on the branches below.

| Branch | Contents |
|---|---|
| [regression-failure-case](https://github.com/srushtinannaware/tfm_failure_cases/tree/regression-failure-case) | Wide/short regression, arithmetic supporting experiments, and GSE40279 real-world validation |
| [l3-xor-failure-case](https://github.com/srushtinannaware/tfm_failure_cases/tree/l3-xor-failure-case) | Multi-condition XOR and irrelevant-feature controls |
| [final_pipeline](https://github.com/srushtinannaware/tfm_failure_cases/tree/final_pipeline) | Random context routing and feature-grouping experiments |
| [models](https://github.com/srushtinannaware/tfm_failure_cases/tree/models) | Shared model adapters and configuration scaffolding |

The historical `pipeline` branch is retained for provenance but is not part of the final implementation or reproduction instructions.

## Reproducing the experiments

Each experiment branch contains its own environment definition, commands, saved raw results, and plotting instructions. This separation is intentional: some model packages have conflicting dependency and hardware requirements.

Start by checking out the relevant branch:

```bash
git clone https://github.com/srushtinannaware/tfm_failure_cases.git
cd tfm_failure_cases
git switch regression-failure-case
```

Then follow that branch's README. Pretrained model checkpoints may be downloaded on first use. Exact runtimes depend on hardware, package versions, and model checkpoint availability.

## Evaluation principles

- All models within a benchmark cell receive the same generated dataset and split.
- Random seeds are recorded in the result files.
- Results distinguish exploratory single-seed sweeps from replicated comparisons.
- A poor result is treated as a comparative failure only when another model demonstrates that the same task is learnable.
- Reported mechanisms are hypotheses unless supported by a controlled architectural ablation.

## Scope and limitations

- The benchmarks are synthetic and target specific stress conditions.
- The regression headline uses three seeds; some supporting experiments use one seed.
- Keeping training rows fixed while increasing columns changes both absolute width and the feature-to-row ratio.
- Model packages and pretrained checkpoints evolve, so future versions may not reproduce the exact recorded values.
- The experiments identify behavioral differences but do not establish their architectural causes.

## Repository status

The committed result files document the runs used for the project poster. Numerical results are preserved as recorded, and every reported figure should remain traceable to its generating script and source CSV.

## Citation

If you use this repository, please cite the repository URL and name the relevant experiment branch. A formal archival citation will be added if the project is released through a persistent repository.
