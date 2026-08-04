"""
Turn arithmetic_results.csv into line graphs, one function per experiment
(base column sweep / OOD / depth). Skips a plot with a message rather than
erroring if that experiment hasn't been run yet, so this can be re-run
safely as experiments 2 and 3 get filled in over time.

Usage:
    python plot_arithmetic_results.py --input results/arithmetic_results.csv --outdir results/
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


def plot_experiment1_base(df: pd.DataFrame, task: str, outdir: str):
    """Experiment 1: column sweep, fixed depth, in-distribution."""
    sub = df[(df.experiment == "1_base") & (df.task == task) & (df.status == "ok")].copy()
    if sub.empty:
        print(f"No experiment-1 (base) rows for task={task}; skipping.")
        return

    metric_col, metric_name = METRIC_BY_TASK[task]
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    agg = sub.groupby(["model", "n_features"])[metric_col].mean().reset_index().sort_values("n_features")

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in agg.model.unique():
        m = agg[agg.model == model_key]
        ax.plot(m.n_features, m[metric_col], marker="o", label=LABELS.get(model_key, model_key))
    ax.set_xscale("log")
    ax.set_xlabel("Number of columns (features), rows fixed")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Arithmetic-chain sweep, Experiment 1 (base) — {task} ({metric_name} vs. columns)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out_path = os.path.join(outdir, f"arithmetic_exp1_{task}_{metric_name}_vs_columns.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_experiment2_ood(df: pd.DataFrame, task: str, outdir: str):
    """Experiment 2: column sweep, fixed depth, in-distribution vs. OOD."""
    sub = df[(df.experiment == "2_ood") & (df.task == task) & (df.status == "ok")].copy()
    if sub.empty:
        print(f"No experiment-2 (OOD) rows for task={task} yet; skipping.")
        return

    metric_col, metric_name = METRIC_BY_TASK[task]
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub["ood"] = sub["ood"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in sub.model.unique():
        for ood_flag, style in [("False", "-"), ("True", "--")]:
            m = sub[(sub.model == model_key) & (sub.ood == ood_flag)].sort_values("n_features")
            if m.empty:
                continue
            tag = "OOD" if ood_flag == "True" else "in-dist"
            ax.plot(m.n_features, m[metric_col], style, marker="o", label=f"{LABELS.get(model_key, model_key)} ({tag})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of columns (features), rows fixed")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Arithmetic-chain sweep, Experiment 2 (OOD) — {task}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    out_path = os.path.join(outdir, f"arithmetic_exp2_{task}_{metric_name}_ood_vs_columns.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    if task == "regression":
        _plot_experiment2_regression_readable(sub, metric_col, outdir)


def _plot_experiment2_regression_readable(sub: pd.DataFrame, metric_col: str, outdir: str):
    """Full-range and zoomed views so one extreme value does not hide the rest."""
    zoom_min, zoom_max = -0.4, 0.15
    model_keys = list(sub.model.unique())
    colors = {key: plt.get_cmap("tab10")(idx) for idx, key in enumerate(model_keys)}
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    for model_key in model_keys:
        for ood_flag, style in [("False", "-"), ("True", "--")]:
            m = sub[(sub.model == model_key) & (sub.ood == ood_flag)].sort_values("n_features")
            if m.empty:
                continue
            tag = "OOD" if ood_flag == "True" else "in-dist"
            label = f"{LABELS.get(model_key, model_key)} ({tag})"
            plot_kwargs = {
                "color": colors[model_key],
                "linestyle": style,
                "marker": "o",
                "linewidth": 2,
            }
            ax_full.plot(m.n_features, m[metric_col], label=label, **plot_kwargs)
            ax_zoom.plot(m.n_features, m[metric_col], **plot_kwargs)

            below = m[m[metric_col] < zoom_min]
            for _, row in below.iterrows():
                ax_zoom.scatter(row.n_features, zoom_min, marker="v", s=70, color=colors[model_key], zorder=5)
                ax_zoom.annotate(
                    f"{row[metric_col]:.2f}",
                    (row.n_features, zoom_min),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=colors[model_key],
                )

    for ax in (ax_full, ax_zoom):
        ax.set_xscale("log")
        ax.set_xlabel("Number of columns (features), rows fixed")
        ax.axhline(0, color="black", linewidth=1, alpha=0.45)
        ax.grid(True, which="both", alpha=0.3)

    ax_full.set_ylabel("r2")
    ax_full.set_title("Full range")
    ax_full.legend(fontsize=8, ncol=2)
    ax_zoom.set_ylim(zoom_min, zoom_max)
    ax_zoom.set_yticks([zoom_min, -0.3, -0.2, -0.1, 0.0, 0.1])
    ax_zoom.set_title("Zoomed comparison (-0.4 to 0.15)")
    fig.suptitle("Arithmetic-chain Experiment 2: regression under distribution shift", fontsize=16)

    out_path = os.path.join(outdir, "arithmetic_exp2_regression_r2_ood_readable.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_experiment3_depth(df: pd.DataFrame, task: str, outdir: str):
    """Experiment 3: fixed column count, depth swept, in-distribution."""
    sub = df[(df.experiment == "3_depth") & (df.task == task) & (df.status == "ok")].copy()
    if sub.empty:
        print(f"No experiment-3 (depth) rows for task={task} yet; skipping.")
        return

    metric_col, metric_name = METRIC_BY_TASK[task]
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    agg = sub.groupby(["model", "depth"])[metric_col].mean().reset_index().sort_values("depth")

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in agg.model.unique():
        m = agg[agg.model == model_key]
        ax.plot(m.depth, m[metric_col], marker="o", label=LABELS.get(model_key, model_key))
    ax.set_xlabel("Hierarchical depth (number of chained arithmetic operations)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Arithmetic-chain sweep, Experiment 3 (depth) — {task}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(outdir, f"arithmetic_exp3_{task}_{metric_name}_vs_depth.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/arithmetic_results.csv")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    for task in ["classification", "regression"]:
        plot_experiment1_base(df, task, args.outdir)
        plot_experiment2_ood(df, task, args.outdir)
        plot_experiment3_depth(df, task, args.outdir)


if __name__ == "__main__":
    main()
