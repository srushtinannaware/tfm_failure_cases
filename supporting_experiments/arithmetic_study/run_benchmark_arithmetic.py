"""
Benchmark runner for the arithmetic-chain generator (arithmetic_generation.py).

Separate from run_benchmark.py (which runs the sklearn-based column
sweep) so the two studies stay cleanly independent, while reusing the
same model wrappers (models.py) and per-cell evaluation logic
(run_benchmark.py's _eval_classification / _eval_regression) so results
are directly comparable.

Three named experiments, selected with --experiment:

  1 (base)  Column sweep 50 -> 2,000 (same range as the main study),
            fixed depth=5, in-distribution (no shift). Tests whether the
            arithmetic-chain rule produces the same kind of width-driven
            failure as the sklearn generator.

  2 (ood)   Same column sweep, fixed depth=5, but each cell is run twice
            -- once in-distribution, once with ood=True (shifted test
            set) -- so accuracy/R2 can be compared side by side within
            the same n_features. Tests genuine train/test distribution
            shift, which the main study never covers.

  3 (depth) Fixed column count (default 500), depth swept via
            DEPTH_SWEEP=[5, 10, 20], in-distribution. Tests whether a
            longer chain of sequential operations makes failure worse
            independent of column count -- the "20 steps vs. 5 steps"
            hierarchical-complexity comparison.

Usage:
    python run_benchmark_arithmetic.py                  # experiment 1 (base)
    python run_benchmark_arithmetic.py --experiment 2    # OOD
    python run_benchmark_arithmetic.py --experiment 3    # depth

Single seed only (seed=0 by default) -- no multi-seed runs for this
study, per the same reasoning as the main sweep's results: the seed=0
signal is being treated as notable enough to report directly rather than
averaged over repeats.
"""

from __future__ import annotations

import argparse
import csv
import os

from arithmetic_generation import DEPTH_SWEEP, make_arithmetic_dataset
from data_generation import COLUMN_SWEEP
from models import MODEL_KEYS, get_model, get_tabicl_device, get_tabpfn_device
from run_benchmark import _eval_classification, _eval_regression

RESULTS_COLUMNS = [
    "experiment",
    "task",
    "model",
    "n_features",
    "depth",
    "ood",
    "n_train",
    "n_test",
    "seed",
    "status",
    "metric_primary_name",
    "metric_primary",
    "metric_secondary_name",
    "metric_secondary",
    "fit_seconds",
    "predict_seconds",
    "error",
]


def _run_cell(writer, f, experiment: str, task: str, model_key: str, ds, n_features: int, depth: int, ood: bool, seed: int, n_train: int, n_test: int):
    row = {
        "experiment": experiment,
        "task": task,
        "model": model_key,
        "n_features": n_features,
        "depth": depth,
        "ood": ood,
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "status": "ok",
        "metric_primary_name": "",
        "metric_primary": "",
        "metric_secondary_name": "",
        "metric_secondary": "",
        "fit_seconds": "",
        "predict_seconds": "",
        "error": "",
    }
    label = f"[exp={experiment:5}] [{task:14}] n_features={n_features:>5} depth={depth:>3} ood={ood!s:5} model={model_key:14}"
    import traceback

    try:
        model = get_model(model_key, task)
        metrics = _eval_classification(model, ds) if task == "classification" else _eval_regression(model, ds)
        row.update(metrics)
        print(
            f"{label} OK  "
            f"{metrics['metric_primary_name']}={metrics['metric_primary']:.4f}  "
            f"fit={metrics['fit_seconds']:.2f}s"
        )
    except Exception as e:  # noqa: BLE001 - intentionally broad, logged below
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"{label} ERROR  {type(e).__name__}: {e}")
        traceback.print_exc()

    writer.writerow(row)
    f.flush()


def run_experiment_1_base(writer, f, tasks, model_keys, column_sweep, depth, seed, n_train, n_test):
    """Column sweep, fixed depth, in-distribution."""
    for task in tasks:
        for n_features in column_sweep:
            ds = make_arithmetic_dataset(task, n_features, depth=depth, n_train=n_train, n_test=n_test, seed=seed, ood=False)
            for model_key in model_keys:
                _run_cell(writer, f, "1_base", task, model_key, ds, n_features, depth, False, seed, n_train, n_test)


def run_experiment_2_ood(writer, f, tasks, model_keys, column_sweep, depth, seed, n_train, n_test):
    """Column sweep, fixed depth, in-distribution vs. OOD side by side."""
    for task in tasks:
        for n_features in column_sweep:
            for ood_flag in (False, True):
                ds = make_arithmetic_dataset(task, n_features, depth=depth, n_train=n_train, n_test=n_test, seed=seed, ood=ood_flag)
                for model_key in model_keys:
                    _run_cell(writer, f, "2_ood", task, model_key, ds, n_features, depth, ood_flag, seed, n_train, n_test)


def run_experiment_3_depth(writer, f, tasks, model_keys, n_features, depth_sweep, seed, n_train, n_test):
    """Fixed column count, depth swept, in-distribution."""
    for task in tasks:
        for depth in depth_sweep:
            ds = make_arithmetic_dataset(task, n_features, depth=depth, n_train=n_train, n_test=n_test, seed=seed, ood=False)
            for model_key in model_keys:
                _run_cell(writer, f, "3_depth", task, model_key, ds, n_features, depth, False, seed, n_train, n_test)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["1", "2", "3"], default="1", help="1=base, 2=ood, 3=depth")
    p.add_argument("--tasks", nargs="+", default=["classification", "regression"])
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    p.add_argument("--column-sweep", nargs="+", type=int, default=COLUMN_SWEEP, help="used by experiments 1 and 2")
    p.add_argument("--depth", type=int, default=5, help="fixed depth used by experiments 1 and 2")
    p.add_argument("--n-features", type=int, default=500, help="fixed column count used by experiment 3")
    p.add_argument("--depth-sweep", nargs="+", type=int, default=DEPTH_SWEEP, help="used by experiment 3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--output", default="results/arithmetic_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Only resolves/prints (and imports torch for) the devices actually
    # needed by the requested models -- an xgboost-only run must never
    # touch torch at all, see models.py's device-getter comments.
    device_bits = []
    if any(m in args.models for m in ("tabpfn_v2_5", "tabpfn_v3")):
        device_bits.append(f"TabPFN device: {get_tabpfn_device()}")
    if "tabicl_v2" in args.models:
        device_bits.append(f"TabICL device: {get_tabicl_device()}")
    print(" | ".join(device_bits) if device_bits else "No TabPFN/TabICL models selected")
    print(f"Experiment {args.experiment}\n")

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()  # written to disk immediately, not just buffered --
            # so if a model crashes the process (e.g. a native-library
            # segfault) before any row is written, the header still
            # survives and a later --models run doesn't skip it and leave
            # the CSV headerless.

        if args.experiment == "1":
            run_experiment_1_base(writer, f, args.tasks, args.models, args.column_sweep, args.depth, args.seed, args.n_train, args.n_test)
        elif args.experiment == "2":
            run_experiment_2_ood(writer, f, args.tasks, args.models, args.column_sweep, args.depth, args.seed, args.n_train, args.n_test)
        elif args.experiment == "3":
            run_experiment_3_depth(writer, f, args.tasks, args.models, args.n_features, args.depth_sweep, args.seed, args.n_train, args.n_test)

    print(f"\nDone. Results written to {args.output}")
