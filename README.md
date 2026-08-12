# Shared model registry

This branch documents the model adapters used across the classification experiments. It replaces the earlier empty per-model directory scaffold with a small executable registry.

The regression experiment retains a separate registry because it requires both classifier and regressor APIs, explicit device routing, and model-version behavior specific to the wide/short study.

## Supported classification models

| Command name | Display name | Python package |
|---|---|---|
| `catboost` | CatBoost | `catboost` |
| `realmlp` | RealMLP-TD | `pytabkit` |
| `tabpfn_v2` | TabPFN-v2 | `tabpfn` |
| `tabpfn_v2_5` | TabPFN-v2.5 | `tabpfn` |
| `tabpfn_v3` | TabPFN-v3 | `tabpfn` |
| `tabicl_v2` | TabICL v2 | `tabicl` |

LimiX is omitted from the shared executable registry because it requires a repository checkout, a model-specific configuration file, and a checkpoint installation outside the standard Python package workflow. The XOR branch contains the adapter used for its recorded LimiX runs.

## Installation

```bash
conda env create -f environment.yml
conda activate tfm-failure
```

Model weights may be downloaded when a factory is first instantiated.

## Usage

List registered models:

```bash
python main.py --list
```

Resolve model factories without loading optional dependencies:

```python
from models import get_model_factories

factories = get_model_factories(["catboost", "tabpfn_v3"])
catboost = factories["CatBoost"]()
```

Imports are lazy. A package is imported only when its factory is called, so unavailable optional models do not prevent other experiments from starting.

## Design requirements

Every registered classifier must provide:

- `fit(X_train, y_train)`;
- `predict(X_test)`;
- `predict_proba(X_test)`.

Experiment branches are responsible for recording the selected checkpoint, random seed, hardware, and any model-specific settings that affect a reported result.
