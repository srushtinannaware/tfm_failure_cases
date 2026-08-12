import pytest

from models import available_models, get_model_factories


def test_expected_models_are_registered():
    assert available_models() == (
        "catboost",
        "realmlp",
        "tabicl_v2",
        "tabpfn_v2",
        "tabpfn_v2_5",
        "tabpfn_v3",
    )


def test_aliases_resolve_without_importing_optional_packages():
    factories = get_model_factories(["cat_boost", "tabpfn3"])
    assert tuple(factories) == ("CatBoost", "TabPFN-v3")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_factories(["not-a-model"])
