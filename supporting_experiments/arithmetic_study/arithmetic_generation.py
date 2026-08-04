"""
Synthetic arithmetic-chain dataset generator.

A second generator, structurally different from data_generation.py's
sklearn-based one. There, the labeling rule is a single linear cut
(classification) or a single weighted sum (regression) over the
informative columns -- one step, regardless of column count. Here, the
label is built by chaining `depth` sequential arithmetic operations over
randomly chosen active columns, e.g. depth=5 applies 5 operations,
depth=20 applies 20. This maps directly onto testing whether a TFM fails
differently once the ground-truth rule requires more sequential steps to
reconstruct, independent of column count or noise -- the "20 steps
hierarchical decision making vs. 5 steps" comparison.

Also supports a genuine train/test distribution shift (`ood=True`): train
features are drawn from a standard normal, test features from a
shifted/rescaled normal, so the same arithmetic rule is evaluated on
inputs the model never trained on. This is a true out-of-distribution
test, unlike the main column sweep, where train and test always come from
the same distribution.

IMPORTANT (fixed after an early run surfaced this): the active columns
and the chain are constructed from a "structure" RNG seeded only from
`seed`, never from `n_features`. Earlier this drew the active columns
with `rng.choice(n_features, ...)`, whose output depends on the
population size even for a fixed seed -- so a column sweep was silently
testing a different random rule at every width instead of the same rule
diluted by more noise. That produced a non-monotonic bounce in early
results (accuracy recovering at 2,000 columns after collapsing at 100)
that all three models tracked in lockstep, which is what gave it away --
a real architectural effect should make models diverge, not move
together. Active columns are now the fixed indices 0..n_active-1, and
`n_active` no longer scales with `n_features` (see `make_arithmetic_dataset`),
so the same rule is reused across an entire sweep and only the amount of
surrounding noise changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_generation import Dataset

OPS = ["+", "-", "*", "/", "mean"]

# Hierarchical-depth sweep: how many sequential arithmetic steps the
# ground-truth rule chains together, independent of column count.
DEPTH_SWEEP = [5, 10, 20]

_EPS = 1e-3


@dataclass
class ArithmeticChain:
    """A sequence of (operation, feature_index) steps defining the label rule."""

    seed_idx: int
    steps: list[tuple[str, int]]

    def __call__(self, X_active: np.ndarray) -> np.ndarray:
        result = X_active[:, self.seed_idx].copy()
        for op, feat_idx in self.steps:
            val = X_active[:, feat_idx]
            if op == "+":
                result = result + val
            elif op == "-":
                result = result - val
            elif op == "*":
                result = result * val
            elif op == "/":
                safe_val = np.where(np.abs(val) < _EPS, _EPS, val)
                result = result / safe_val
            elif op == "mean":
                result = (result + val) / 2.0
            else:
                raise ValueError(f"Unknown op: {op}")
        return result


def _build_chain(n_active: int, depth: int, rng: np.random.RandomState) -> ArithmeticChain:
    seed_idx = int(rng.randint(0, n_active))
    steps = [(str(rng.choice(OPS)), int(rng.randint(0, n_active))) for _ in range(depth)]
    return ArithmeticChain(seed_idx=seed_idx, steps=steps)


def _make_features(
    n_samples: int, n_features: int, rng: np.random.RandomState, shift: float = 0.0, scale: float = 1.0
) -> np.ndarray:
    return rng.normal(loc=shift, scale=scale, size=(n_samples, n_features))


def _robust_clip(raw_train: np.ndarray, raw_test: np.ndarray, n_mad: float = 10.0):
    """Clip the chain's raw output to a robust range before it becomes a
    label. The chain includes a "/" op with only a 1e-3 floor on the
    denominator, so a single near-zero draw can spike the result by up to
    ~1000x -- a handful of these outliers then dominate the regression
    target (and swamp R^2) regardless of column count, at every point in
    the sweep, not just at high columns. Classification is unaffected in
    practice since its threshold is the train median, which clipping
    symmetric around the median doesn't move. Bounds are derived from
    train only and applied to both splits, so this is a fixed transform,
    not a leak.
    """
    median = np.median(raw_train)
    mad = np.median(np.abs(raw_train - median)) + _EPS
    lo, hi = median - n_mad * mad, median + n_mad * mad
    return np.clip(raw_train, lo, hi), np.clip(raw_test, lo, hi)


def make_arithmetic_dataset(
    task: str,
    n_features: int,
    depth: int = 5,
    n_active: int = 20,
    n_train: int = 100,
    n_test: int = 500,
    seed: int = 0,
    ood: bool = False,
    ood_shift: float = 2.0,
    ood_scale: float = 1.5,
) -> Dataset:
    """Build a dataset whose label is a `depth`-step arithmetic chain over
    `n_active` columns; the remaining columns are pure noise, uninvolved
    in the label.

    `n_active` is a fixed count (default 20), NOT scaled with
    `n_features` -- the active columns are always indices
    0..n_active-1, and the chain is built from a structure RNG seeded
    only from `seed`. That means calling this with the same `seed` and
    `depth` across a column sweep reuses the exact same rule every time;
    only the amount of surrounding noise (n_features - n_active) grows.
    This is what makes the column sweep a genuine "same rule, more
    dilution" comparison instead of a fresh random rule at every width.

    If `ood=True`, test-set features are drawn from a shifted/rescaled
    normal distribution instead of the training distribution, so the
    arithmetic rule is evaluated on genuinely out-of-distribution inputs
    at test time -- train and test no longer come from the same
    distribution, unlike every dataset in data_generation.py.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task: {task}")

    n_active = min(n_active, n_features)

    # Structure RNG: seeded only from `seed`, independent of n_features,
    # so the same active columns and the same chain are reused at every
    # point in a column sweep. Active columns are fixed as the first
    # n_active indices -- which specific indices count as "active" is
    # arbitrary since columns are otherwise exchangeable, so pinning them
    # removes n_features-dependent randomness without losing generality.
    structure_rng = np.random.RandomState(seed)
    active_idx = np.arange(n_active)
    chain = _build_chain(n_active, depth, structure_rng)

    # Data RNG: a separate, decorrelated stream for the actual feature
    # values and label noise, so changing n_features only adds/removes
    # noise columns rather than reshuffling the rule itself.
    data_rng = np.random.RandomState(seed + 1_000_003)
    X_train = _make_features(n_train, n_features, data_rng)
    X_test = (
        _make_features(n_test, n_features, data_rng, shift=ood_shift, scale=ood_scale)
        if ood
        else _make_features(n_test, n_features, data_rng)
    )

    raw_train = chain(X_train[:, active_idx])
    raw_test = chain(X_test[:, active_idx])
    raw_train, raw_test = _robust_clip(raw_train, raw_test)

    if task == "regression":
        noise_scale = 0.1 * (np.std(raw_train) + _EPS)
        y_train = raw_train + data_rng.normal(0, noise_scale, size=n_train)
        y_test = raw_test + data_rng.normal(0, noise_scale, size=n_test)
    else:  # classification
        threshold = np.median(raw_train)
        y_train = (raw_train > threshold).astype(int)
        y_test = (raw_test > threshold).astype(int)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
        task=task,
        seed=seed,
    )


if __name__ == "__main__":
    print("In-distribution:")
    for depth in DEPTH_SWEEP:
        ds = make_arithmetic_dataset("classification", n_features=200, depth=depth, seed=0)
        balance = ds.y_train.mean()
        print(f"  depth={depth:>3}  X_train={ds.X_train.shape}  class balance={balance:.2f}")

    print("Out-of-distribution (shifted test set):")
    for depth in DEPTH_SWEEP:
        ds = make_arithmetic_dataset("classification", n_features=200, depth=depth, seed=0, ood=True)
        balance = ds.y_test.mean()
        print(f"  depth={depth:>3}  test class balance={balance:.2f}  (train threshold applied to shifted test)")
