"""
Unified model wrappers for the failure-case comparison.

Scope:

    - TabPFN-2.5    (Nov 2025 release; design ceiling: 100K rows x 2,000 features)
    - TabPFN-3      (May 2026 release; row/feature trade-off, default regime
                     favors more rows over more columns -- 1M x 200, 100K x 2K,
                     or 1K x 20K)
    - TabICL v2     (soda-inria; pretrained on 2-100 columns, claims but does
                     not guarantee generalization beyond that)
    - XGBoost       (classical gradient-boosted trees; not a tabular
                     foundation model at all -- included as a baseline
                     reference so "TFM accuracy dropped" can be read
                     against "and here's what a non-TFM does in the same
                     regime," rather than only compared to the other TFMs)

TabPFN v2 (the original Nature release) and a Random Forest baseline were
in an earlier draft of this scope but are parked for later -- not part of
this comparison right now.

All wrappers expose the same tiny interface:

    model = get_model("tabpfn_v3", task="classification")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)          # classification: labels
    proba = model.predict_proba(X_test)    # classification only

so run_benchmark.py can loop over model names without caring about the
underlying library's API differences.

IMPORTANT: TabPFN and TabICL both download pretrained checkpoints from
Hugging Face Hub on first `.fit()` call. If your environment blocks
huggingface.co, construction succeeds but `.fit()` will raise a
connection/HTTP error. See SETUP.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Feature counts beyond which each checkpoint's own guardrail would refuse to
# run unless explicitly overridden. We override on purpose -- that's the
# whole point of the failure-case study: pushing past the design ceiling and
# observing what actually happens to accuracy, not just whether it errors.
IGNORE_PRETRAINING_LIMITS = True


def _best_available_device() -> str:
    """Best available torch device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# TabPFN and TabICL get separate device choices on purpose -- they don't
# behave the same way under memory pressure, so one blanket "use GPU" or
# "use CPU" decision doesn't fit both.
#
# Both are resolved lazily (on first use, then cached) instead of as eager
# module-level constants. Resolving TabPFN's device eagerly means calling
# _best_available_device(), which imports torch -- and that would happen
# on every `import models`, even in a process that only ever wants
# xgboost. Loading torch and xgboost into the same process reliably
# segfaults on macOS (see _xgboost_model's comment). Keeping this lazy
# means an xgboost-only run never imports torch in the first place.
_tabpfn_device_cache: str | None = None
_tabicl_device_cache: str | None = None


def get_tabpfn_device() -> str:
    """TabPFN's device, resolved on first call and cached.

    Its per-estimator feature subsampling caps how many columns any
    single ensemble member ever processes at once (~150-200, regardless of
    total column count -- see AUTO_FEATURE_SUBSAMPLING_TOP_K in tabpfn's own
    source). That structurally bounds its memory use even at 5,000 columns,
    and it never once OOM'd during testing. Safe to use the fastest device
    available.
    """
    global _tabpfn_device_cache
    if _tabpfn_device_cache is None:
        _tabpfn_device_cache = os.environ.get("FAILURE_CASE_TABPFN_DEVICE") or _best_available_device()
    return _tabpfn_device_cache


def get_tabicl_device() -> str:
    """TabICL's device, resolved on first call and cached.

    No equivalent cap to TabPFN's. Its column-attention cost scales with total
    feature count, and it crashed with "MPS backend out of memory" at 1,000
    columns during testing -- even after tuning batch_size down to 1, the
    minimum. CPU has no equivalent per-device memory ceiling, so it trades
    speed for actually completing the full sweep without crashing.
    """
    global _tabicl_device_cache
    if _tabicl_device_cache is None:
        _tabicl_device_cache = os.environ.get("FAILURE_CASE_TABICL_DEVICE") or "cpu"
    return _tabicl_device_cache


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str  # "tabpfn" | "tabicl" | "baseline"


MODEL_SPECS = [
    ModelSpec("tabpfn_v2_5", "TabPFN-2.5", "tabpfn"),
    ModelSpec("tabpfn_v3", "TabPFN-3", "tabpfn"),
    ModelSpec("tabicl_v2", "TabICL v2", "tabicl"),
    ModelSpec("xgboost", "XGBoost", "baseline"),
]

MODEL_KEYS = [m.key for m in MODEL_SPECS]


class ModelUnavailableError(RuntimeError):
    """Raised when a model can't be constructed or fitted (e.g. missing
    checkpoint download, blocked network, missing license token)."""


def _tabpfn_model(version: str, task: str):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_map = {
        "tabpfn_v2_5": ModelVersion.V2_5,
        "tabpfn_v3": ModelVersion.V3,
    }
    mv = version_map[version]

    common_kwargs: dict[str, Any] = {"device": get_tabpfn_device()}
    if IGNORE_PRETRAINING_LIMITS:
        common_kwargs["ignore_pretraining_limits"] = True

    if task == "classification":
        return TabPFNClassifier.create_default_for_version(mv, **common_kwargs)
    elif task == "regression":
        return TabPFNRegressor.create_default_for_version(mv, **common_kwargs)
    raise ValueError(f"Unknown task: {task}")


def _tabicl_model(task: str):
    from tabicl import TabICLClassifier, TabICLRegressor

    # batch_size=1 processes the 8-member ensemble one estimator at a time
    # instead of all 8 concurrently -- same ensemble, same accuracy, just
    # trades speed for peak memory. Kept on regardless of device, since
    # holding all 8 members' tensors at once is worth avoiding on
    # memory-constrained machines even on CPU, not just on MPS where it
    # can cause an out-of-memory crash.
    tabicl_kwargs: dict[str, Any] = {"device": get_tabicl_device(), "batch_size": 1}

    # checkpoint_version left at its package default, which at the time of
    # writing is the "v2" checkpoint (tabicl-classifier-v2-*.ckpt /
    # tabicl-regressor-v2-*.ckpt) -- i.e. TabICL v2.
    if task == "classification":
        return TabICLClassifier(**tabicl_kwargs)
    elif task == "regression":
        return TabICLRegressor(**tabicl_kwargs)
    raise ValueError(f"Unknown task: {task}")


def _xgboost_model(task: str):
    # IMPORTANT: xgboost and torch (used by tabpfn/tabicl via MPS) each
    # bundle their own OpenMP runtime, and loading both into one process
    # then doing real MPS work reliably segfaults on macOS -- with no
    # Python-catchable exception, regardless of import order. Run xgboost
    # in a separate process from tabpfn/tabicl (e.g. two invocations of
    # run_benchmark_arithmetic.py filtered by --models), never in the same
    # one. Don't move this import to module level -- that would load
    # xgboost even for tabpfn/tabicl-only runs and reintroduce the crash.
    from xgboost import XGBClassifier, XGBRegressor

    # Deliberately regularized rather than left at library defaults: with
    # only 100 training rows and up to thousands of columns, an
    # untuned/deep XGBoost would just overfit or take the comparison out
    # of good faith. max_depth=4 and n_estimators=200 keep trees shallow
    # and the ensemble small; colsample_bytree=0.5 makes each tree see
    # only half the columns -- a direct, if cruder, analogue to TabPFN's
    # per-estimator feature subsampling, so the two aren't being compared
    # on wildly different "how much of the input do you even look at"
    # terms. No GPU needed; runs on CPU regardless of TABPFN_DEVICE /
    # TABICL_DEVICE.
    common_kwargs: dict[str, Any] = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=0,
    )

    if task == "classification":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common_kwargs)
    elif task == "regression":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common_kwargs)
    raise ValueError(f"Unknown task: {task}")


def get_model(model_key: str, task: str):
    """Construct a fresh, unfitted model instance for the given task.

    Raises ModelUnavailableError if the underlying library isn't installed.
    Network/checkpoint-download failures surface later, at `.fit()` time,
    as whatever exception the underlying library raises (usually an
    huggingface_hub / httpx connection error) -- run_benchmark.py catches
    those explicitly so one blocked model doesn't kill the whole sweep.
    """
    try:
        if model_key in ("tabpfn_v2_5", "tabpfn_v3"):
            return _tabpfn_model(model_key, task)
        elif model_key == "tabicl_v2":
            return _tabicl_model(task)
        elif model_key == "xgboost":
            return _xgboost_model(task)
        raise ValueError(f"Unknown model_key: {model_key}")
    except ImportError as e:
        raise ModelUnavailableError(f"{model_key}: required package not installed ({e})") from e


if __name__ == "__main__":
    # Construction-only smoke test (no network needed).
    print(f"TabPFN device: {get_tabpfn_device()}")
    print(f"TabICL device: {get_tabicl_device()}")
    for spec in MODEL_SPECS:
        for task in ["classification", "regression"]:
            m = get_model(spec.key, task)
            print(f"{spec.key:14} {task:15} -> {type(m).__name__}")
