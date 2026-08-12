"""Lazy factories for the classification models used in this project."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


ModelFactory = Callable[[], Any]


class ModelSetupError(RuntimeError):
    """Raised when a selected optional model cannot be constructed."""


def _tabpfn_factory(version_member: str) -> ModelFactory:
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
                f"The installed TabPFN package does not provide {version_member}."
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


_REGISTRY: dict[str, tuple[str, ModelFactory]] = {
    "catboost": ("CatBoost", _create_catboost),
    "realmlp": ("RealMLP-TD", _create_realmlp),
    "tabicl_v2": ("TabICL-v2", _create_tabicl_v2),
    "tabpfn_v2": ("TabPFN-v2", _tabpfn_factory("V2")),
    "tabpfn_v2_5": ("TabPFN-v2.5", _tabpfn_factory("V2_5")),
    "tabpfn_v3": ("TabPFN-v3", _tabpfn_factory("V3")),
}

_ALIASES = {
    "cat_boost": "catboost",
    "real_mlp": "realmlp",
    "realmlp_td": "realmlp",
    "tabicl": "tabicl_v2",
    "tabpfn2": "tabpfn_v2",
    "tabpfn25": "tabpfn_v2_5",
    "tabpfn2_5": "tabpfn_v2_5",
    "tabpfn3": "tabpfn_v3",
}


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(".", "_")
    return _ALIASES.get(normalized, normalized)


def available_models() -> tuple[str, ...]:
    """Return canonical command-line names in registry order."""
    return tuple(_REGISTRY)


def get_model_factories(model_names: Iterable[str]) -> dict[str, ModelFactory]:
    """Resolve names without importing optional model packages."""
    selected: dict[str, ModelFactory] = {}
    unknown: list[str] = []

    for requested_name in model_names:
        canonical_name = _normalize_name(requested_name)
        entry = _REGISTRY.get(canonical_name)
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
