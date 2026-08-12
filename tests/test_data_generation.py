import numpy as np

from data_generation import make_wide_short_classification, make_wide_short_regression


def test_regression_generation_is_reproducible():
    first = make_wide_short_regression(200, n_train=100, n_test=50, seed=7)
    second = make_wide_short_regression(200, n_train=100, n_test=50, seed=7)

    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_regression_split_shapes():
    dataset = make_wide_short_regression(200, n_train=100, n_test=500, seed=0)

    assert dataset.X_train.shape == (100, 200)
    assert dataset.X_test.shape == (500, 200)
    assert dataset.y_train.shape == (100,)
    assert dataset.y_test.shape == (500,)


def test_classification_split_shapes_and_labels():
    dataset = make_wide_short_classification(100, n_train=100, n_test=50, seed=0)

    assert dataset.X_train.shape == (100, 100)
    assert dataset.X_test.shape == (50, 100)
    assert set(np.unique(dataset.y_train)).issubset({0, 1})
    assert set(np.unique(dataset.y_test)).issubset({0, 1})
