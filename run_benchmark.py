"""
Benchmark runner for the wide-short failure-case study.

Sweeps column count (rows fixed at n_train, default 100) across models and
seeds, records accuracy/AUC (classification) or R2/RMSE (regression) plus
fit/predict wall-clock time, and writes results incrementally to a CSV so a
long run can be interrupted without losing progress.

Usage:
    python run_benchmark.py
    python run_benchmark.py --models tabicl_v2 tabpfn_v2_5 --seeds 0 1
    python run_benchmark.py --column-sweep 50 200 1000 5000 --tasks classification

Each (task, n_features, seed, model) cell is run independently and wrapped
in a broad try/except: if a model can't be constructed or fitted (e.g. a
blocked network preventing a Hugging Face checkpoint download, or a missing
license token -- see SETUP.md), that cell is recorded with status="error"
and the sweep continues rather than aborting.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

from data_generation import COLUMN_SWEEP, make_dataset
from models import MODEL_KEYS, get_model, get_tabicl_device, get_tabpfn_device

RESULTS_COLUMNS = [
    "task",
    "model",
    "n_features",
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


def _eval_classification(model, ds) -> dict:
    t0 = time.perf_counter()
    model.fit(ds.X_train, ds.y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(ds.X_test)
    predict_s = time.perf_counter() - t0

    acc = accuracy_score(ds.y_test, preds)
    auc = float("nan")
    try:
        proba = model.predict_proba(ds.X_test)
        if proba.shape[1] == 2:
            auc = roc_auc_score(ds.y_test, proba[:, 1])
    except Exception:
        pass

    return {
        "metric_primary_name": "accuracy",
        "metric_primary": acc,
        "metric_secondary_name": "roc_auc",
        "metric_secondary": auc,
        "fit_seconds": fit_s,
        "predict_seconds": predict_s,
    }


def _eval_regression(model, ds) -> dict:
    t0 = time.perf_counter()
    model.fit(ds.X_train, ds.y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(ds.X_test)
    predict_s = time.perf_counter() - t0

    r2 = r2_score(ds.y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(ds.y_test, preds)))

    return {
        "metric_primary_name": "r2",
        "metric_primary": r2,
        "metric_secondary_name": "rmse",
        "metric_secondary": rmse,
        "fit_seconds": fit_s,
        "predict_seconds": predict_s,
    }


def run(
    tasks: list[str],
    model_keys: list[str],
    column_sweep: list[int],
    seeds: list[int],
    n_train: int,
    n_test: int,
    output_csv: str,
):
    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()

        for task in tasks:
            for n_features in column_sweep:
                for seed in seeds:
                    ds = make_dataset(task, n_features, n_train=n_train, n_test=n_test, seed=seed)
                    for model_key in model_keys:
                        row = {
                            "task": task,
                            "model": model_key,
                            "n_features": n_features,
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
                        label = f"[{task:14}] n_features={n_features:>5} seed={seed} model={model_key:14}"
                        try:
                            model = get_model(model_key, task)
                            if task == "classification":
                                metrics = _eval_classification(model, ds)
                            else:
                                metrics = _eval_regression(model, ds)
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["classification", "regression"])
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    p.add_argument("--column-sweep", nargs="+", type=int, default=COLUMN_SWEEP)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--output", default="results/results.csv")
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
    print((" | ".join(device_bits) if device_bits else "No TabPFN/TabICL models selected") + "\n")
    run(
        tasks=args.tasks,
        model_keys=args.models,
        column_sweep=args.column_sweep,
        seeds=args.seeds,
        n_train=args.n_train,
        n_test=args.n_test,
        output_csv=args.output,
    )
    print(f"\nDone. Results written to {args.output}")
