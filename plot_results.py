"""
Turn results.csv into the line graphs:
performance vs. number of columns, one line per model, at fixed n_train rows.

Usage:
    python plot_results.py --input results/results.csv --outdir results/
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models import MODEL_SPECS

LABELS = {m.key: m.label for m in MODEL_SPECS}
METRIC_BY_TASK = {
    "classification": ("metric_primary", "accuracy"),
    "regression": ("metric_primary", "r2"),
}


def plot_task(df: pd.DataFrame, task: str, outdir: str):
    sub = df[(df.task == task) & (df.status == "ok")].copy()
    if sub.empty:
        print(f"No successful runs for task={task}; skipping plot.")
        return

    metric_col, metric_name = METRIC_BY_TASK[task]
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")

    agg = (
        sub.groupby(["model", "n_features"])[metric_col]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("n_features")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in agg.model.unique():
        m = agg[agg.model == model_key]
        label = LABELS.get(model_key, model_key)
        ax.plot(m.n_features, m["mean"], marker="o", label=label)
        ax.fill_between(
            m.n_features,
            m["mean"] - m["std"].fillna(0),
            m["mean"] + m["std"].fillna(0),
            alpha=0.15,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of columns (features), rows fixed")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Wide/short failure-case sweep — {task} ({metric_name} vs. columns)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out_path = os.path.join(outdir, f"{task}_{metric_name}_vs_columns.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_single_timing(agg: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in agg.model.unique():
        m = agg[agg.model == model_key]
        ax.plot(m.n_features, m[value_col], marker="o", label=LABELS.get(model_key, model_key))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of columns (features), rows fixed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_timing(df: pd.DataFrame, outdir: str):
    sub = df[df.status == "ok"].copy()
    if sub.empty:
        return
    sub["fit_seconds"] = pd.to_numeric(sub["fit_seconds"], errors="coerce")
    sub["predict_seconds"] = pd.to_numeric(sub["predict_seconds"], errors="coerce")

    fit_agg = sub.groupby(["model", "n_features"])["fit_seconds"].mean().reset_index().sort_values("n_features")
    _plot_single_timing(
        fit_agg,
        "fit_seconds",
        "Fit time (seconds)",
        "Fit time vs. columns",
        os.path.join(outdir, "fit_time_vs_columns.png"),
    )

    predict_agg = (
        sub.groupby(["model", "n_features"])["predict_seconds"].mean().reset_index().sort_values("n_features")
    )
    _plot_single_timing(
        predict_agg,
        "predict_seconds",
        "Predict time (seconds)",
        "Predict time vs. columns",
        os.path.join(outdir, "predict_time_vs_columns.png"),
    )

    # Combined fit+predict panel, one figure, two subplots sharing the x-axis --
    # makes it obvious at a glance that TabICL's cost is a predict-time problem,
    # not a fit-time problem (its fit stays cheap; predict blows up at high
    # column counts due to its O(n^2 + n*m^2) column-attention mechanism).
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, agg, value_col, ylabel, title in [
        (axes[0], fit_agg, "fit_seconds", "Fit time (seconds)", "Fit time vs. columns"),
        (axes[1], predict_agg, "predict_seconds", "Predict time (seconds)", "Predict time vs. columns"),
    ]:
        for model_key in agg.model.unique():
            m = agg[agg.model == model_key]
            ax.plot(m.n_features, m[value_col], marker="o", label=LABELS.get(model_key, model_key))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of columns (features), rows fixed")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    out_path = os.path.join(outdir, "timing_vs_columns.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/results.csv")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    for task in ["classification", "regression"]:
        plot_task(df, task, args.outdir)
    plot_timing(df, args.outdir)


if __name__ == "__main__":
    main()
