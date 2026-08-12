import numpy as np

from datasets.logical_dataset import generate_logical_dataset, recompute_target


def test_l3_rule_matches_generated_labels():
    X, y = generate_logical_dataset("L3", n_noise_features=4, seed=2)

    expected = recompute_target(X[:, :4], "L3")
    np.testing.assert_array_equal(y, expected)


def test_noise_features_do_not_change_the_target_for_a_fixed_seed():
    X_clean, y_clean = generate_logical_dataset("L3", n_noise_features=0, seed=3)
    X_noisy, y_noisy = generate_logical_dataset("L3", n_noise_features=8, seed=3)

    np.testing.assert_array_equal(X_clean, X_noisy[:, :4])
    np.testing.assert_array_equal(y_clean, y_noisy)


def test_dataset_generation_is_reproducible():
    first_X, first_y = generate_logical_dataset("L3", n_noise_features=2, seed=4)
    second_X, second_y = generate_logical_dataset("L3", n_noise_features=2, seed=4)

    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
