import sys
from pathlib import Path

import numpy as np

ARITHMETIC_DIR = Path(__file__).parents[1] / "supporting_experiments" / "arithmetic_study"
sys.path.insert(0, str(ARITHMETIC_DIR))

from arithmetic_generation import make_arithmetic_dataset  # noqa: E402


def test_arithmetic_generation_is_reproducible_within_a_width():
    first = make_arithmetic_dataset("regression", n_features=200, depth=5, seed=3)
    second = make_arithmetic_dataset("regression", n_features=200, depth=5, seed=3)

    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_ood_changes_test_distribution_only():
    in_distribution = make_arithmetic_dataset(
        "classification", n_features=50, depth=5, seed=4, ood=False
    )
    shifted = make_arithmetic_dataset(
        "classification", n_features=50, depth=5, seed=4, ood=True
    )

    np.testing.assert_array_equal(in_distribution.X_train, shifted.X_train)
    np.testing.assert_array_equal(in_distribution.y_train, shifted.y_train)
    assert shifted.X_test.mean() > in_distribution.X_test.mean() + 1.0
