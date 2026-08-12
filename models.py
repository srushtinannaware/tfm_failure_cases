"""Lazy model registry for classification benchmarks.

Heavy optional dependencies are imported only when their model is selected.
Every factory returns an estimator implementing ``fit``, ``predict`` and
``predict_proba``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


ModelFactory = Callable[[], Any]


class ModelSetupError(RuntimeError):
    """Raised when a selected model is not installed or configured."""


def _tabpfn_factory(version_member: str) -> ModelFactory:
    """Create a factory pinned to one TabPFN checkpoint generation."""

    def create() -> Any:
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as error:
            raise ModelSetupError(
                "TabPFN is not installed. Install it with: pip install tabpfn"
            ) from error

        try:
            version = getattr(ModelVersion, version_member)
        except AttributeError as error:
            raise ModelSetupError(
                f"The installed TabPFN does not provide {version_member}. "
                "Upgrade it with: pip install --upgrade tabpfn"
            ) from error

        return TabPFNClassifier.create_default_for_version(version)

    return create


def _create_tabicl_v2() -> Any:
    try:
        from tabicl import TabICLClassifier
    except ImportError as error:
        raise ModelSetupError(
            "TabICL is not installed. Install it with: pip install tabicl"
        ) from error

    return TabICLClassifier(
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        random_state=42,
    )


def _create_realmlp() -> Any:
    try:
        from pytabkit import RealMLP_TD_Classifier
    except ImportError as error:
        raise ModelSetupError(
            "RealMLP is not installed. Install it with: pip install pytabkit"
        ) from error

    return RealMLP_TD_Classifier(random_state=42)


def _create_catboost() -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise ModelSetupError(
            "CatBoost is not installed. Install it with: pip install catboost"
        ) from error

    return CatBoostClassifier(random_state=42, verbose=False)


# LimiX is loaded from a local repository checkout.
LIMIX_REPO_PATH = os.environ.get("LIMIX_REPO_PATH", "")
LIMIX_HF_REPO_ID = "stableai-org/LimiX-16M"
LIMIX_CKPT_FILENAME = "LimiX-16M.ckpt"


class _LimiXClassifierAdapter:
    """Adapts LimiX's one-shot predict(X_train, y_train, X_test, task_type)
    interface to the fit/predict/predict_proba shape that
    metrics.evaluate_classifier() expects. LimiX does in-context learning
    like TabPFN, so "fit" just stores the training data; the actual
    inference call happens lazily and is cached so predict() and
    predict_proba() on the same X_test don't re-run inference twice."""

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._cache_key: int | None = None
        self._cache_proba: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LimiXClassifierAdapter":
        self._X_train = X
        self._y_train = y
        self._cache_key = None
        return self

    def _proba(self, X: np.ndarray) -> np.ndarray:
        key = id(X)
        if key != self._cache_key:
            result = self._predictor.predict(
                self._X_train, self._y_train, X, task_type="Classification"
            )
            self._cache_proba = np.asarray(
                result.to("cpu").numpy() if hasattr(result, "to") else result
            )
            self._cache_key = key
        return self._cache_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X).argmax(axis=1)


def _create_limix() -> Any:
    if not LIMIX_REPO_PATH:
        raise ModelSetupError(
            "LimiX isn't on PyPI. Clone "
            "https://github.com/limix-ldm-ai/LimiX, install its "
            "dependencies (torch, flash_attn, huggingface-hub, etc. - see "
            "its README), then set LIMIX_REPO_PATH to that folder before "
            "running, e.g.: export LIMIX_REPO_PATH=/path/to/LimiX"
        )

    try:
        import torch
        from huggingface_hub import hf_hub_download

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        if LIMIX_REPO_PATH not in sys.path:
            sys.path.insert(0, LIMIX_REPO_PATH)

        from inference.predictor import LimiXPredictor
    except ImportError as error:
        raise ModelSetupError(
            "LimiX's dependencies aren't fully installed in this "
            "environment. Follow the manual install steps in the LimiX "
            "README (torch, the pinned flash_attn wheel, huggingface-hub, "
            "scikit-learn, etc.) inside your conda env, then try again."
        ) from error

    checkpoint_path = hf_hub_download(
        repo_id=LIMIX_HF_REPO_ID,
        filename=LIMIX_CKPT_FILENAME,
        local_dir=os.path.join(LIMIX_REPO_PATH, "cache"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The no-retrieval configuration has lower memory requirements.
    inference_config = os.path.join(
        LIMIX_REPO_PATH, "config", "cls_default_noretrieval.json"
    )

    predictor = LimiXPredictor(
        device=device,
        model_path=checkpoint_path,
        inference_config=inference_config,
    )
    return _LimiXClassifierAdapter(predictor)


_MODEL_REGISTRY: dict[str, tuple[str, ModelFactory]] = {
    "tabpfn_v2": ("TabPFN-v2", _tabpfn_factory("V2")),
    "tabpfn_v2_5": ("TabPFN-v2.5", _tabpfn_factory("V2_5")),
    "tabpfn_v3": ("TabPFN-v3", _tabpfn_factory("V3")),
    "tabicl_v2": ("TabICL-v2", _create_tabicl_v2),
    "realmlp": ("RealMLP-TD", _create_realmlp),
    "catboost": ("CatBoost", _create_catboost),
    "limix": ("LimiX-16M", _create_limix),
}

_ALIASES = {
    "tabpfn2": "tabpfn_v2",
    "tabpfn25": "tabpfn_v2_5",
    "tabpfn2_5": "tabpfn_v2_5",
    "tabpfn3": "tabpfn_v3",
    "tabicl": "tabicl_v2",
    "real_mlp": "realmlp",
    "realmlp_td": "realmlp",
    "cat_boost": "catboost",
    "limix_16m": "limix",
}


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(".", "_")
    return _ALIASES.get(normalized, normalized)


def available_models() -> tuple[str, ...]:
    """Return the canonical command-line names in registry order."""
    return tuple(_MODEL_REGISTRY)


def get_model(model_names: Iterable[str]) -> dict[str, ModelFactory]:
    """Resolve requested names to display-name/factory pairs.

    Imports and model-weight downloads occur later, when a returned factory is
    called. Duplicate aliases are rejected to avoid running a model twice.
    """
    selected: dict[str, ModelFactory] = {}
    unknown: list[str] = []

    for requested_name in model_names:
        canonical_name = _normalize_name(requested_name)
        entry = _MODEL_REGISTRY.get(canonical_name)
        if entry is None:
            unknown.append(requested_name)
            continue

        display_name, factory = entry
        if display_name in selected:
            raise ValueError(f"Model requested more than once: {display_name}")
        selected[display_name] = factory

    if unknown:
        choices = ", ".join(available_models())
        raise ValueError(
            f"Unknown model name(s): {', '.join(unknown)}. Available: {choices}"
        )
    if not selected:
        raise ValueError("Select at least one model.")

    return selected
