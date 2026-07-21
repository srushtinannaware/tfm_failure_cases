"""Lazy model registry for classification benchmarks.

Heavy optional dependencies are imported only when their model is selected.
Every factory returns an estimator implementing ``fit``, ``predict`` and
``predict_proba``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


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


def _create_tabfm() -> Any:
    try:
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch
    except ImportError as error:
        raise ModelSetupError(
            "TabFM is not distributed on PyPI. Clone its official repository "
            "and install the PyTorch extra with: pip install -e .[pytorch]"
        ) from error

    backend_model = tabfm_v1_0_0_pytorch.load()
    return TabFMClassifier(model=backend_model)


def _create_nanotabpfn() -> Any:
    raise ModelSetupError(
        "nanoTabPFN is an educational pretraining implementation, not a "
        "ready-to-use pretrained package. Clone automl/nanoTabPFN, train a "
        "NanoTabPFNModel, and add a project-specific factory that constructs "
        "NanoTabPFNClassifier(trained_model, device)."
    )


def _create_realmlp() -> Any:
    try:
        from pytabkit import RealMLP_TD_Classifier
    except ImportError as error:
        raise ModelSetupError(
            "RealMLP is not installed. Install it with: pip install pytabkit"
        ) from error

    return RealMLP_TD_Classifier(random_state=42)


_MODEL_REGISTRY: dict[str, tuple[str, ModelFactory]] = {
    "tabpfn_v2": ("TabPFN-v2", _tabpfn_factory("V2")),
    "tabpfn_v2_5": ("TabPFN-v2.5", _tabpfn_factory("V2_5")),
    "tabpfn_v3": ("TabPFN-v3", _tabpfn_factory("V3")),
    "tabicl_v2": ("TabICL-v2", _create_tabicl_v2),
    "tabfm": ("TabFM-v1.0", _create_tabfm),
    "nanotabpfn": ("nanoTabPFN", _create_nanotabpfn),
    "realmlp": ("RealMLP-TD", _create_realmlp),
}

_ALIASES = {
    "tabpfn2": "tabpfn_v2",
    "tabpfn25": "tabpfn_v2_5",
    "tabpfn2_5": "tabpfn_v2_5",
    "tabpfn3": "tabpfn_v3",
    "tabicl": "tabicl_v2",
    "tabfm_v1": "tabfm",
    "nano_tabpfn": "nanotabpfn",
    "real_mlp": "realmlp",
    "realmlp_td": "realmlp",
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