"""
General results visualizer - works for any dataset results CSV.
Generates accuracy + ROC AUC charts and summary tables.

Run from the project root:
    python3 visualize_results.py --results results/many_class_check_results.csv
    python3 visualize_results.py --results results/context_length_check_results.csv
    python3 visualize_results.py --results results/complex_logical_interaction_results.csv

Or run for ALL result files at once:
    python3 visualize_results.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

import pandas as pd


# ── config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

# maps csv filename → nice title for charts
DATASET_TITLES = {
    "kate_datasets_results.csv":                   "Kate Datasets",
    "many_class_results.csv":                "Many-Class Classification",
    "context_length_check_results.csv":            "Feature Attention Dilution",
    "complex_logical_interaction_results.csv":     "Complex Logical Interaction",
    "periodic_boundary_check_results.csv":         "Periodic Boundary",
    "prior_distribution_check_results.csv":        "Prior Distribution Mismatch",
    "causal_shortcut_check_results.csv":           "Causal Shortcut (S1/S2)",
    "deep_causal_chain_check_results.csv":         "Deep Causal Chain",
    "feature_scale_mismatch_results.csv":          "Feature Scale Mismatch",
}


def _resolve_results_path(results_path: Path) -> Path | None:
    if results_path.exists():
        return results_path

    candidate = results_path
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        return candidate

    if not results_path.is_absolute():
        alt_candidate = RESULTS_DIR / results_path.name
        if alt_candidate.exists():
            return alt_candidate

    stem = results_path.stem
    for suffix in ("_check_results", "_results", "_summary"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    matches = [
        path
        for path in RESULTS_DIR.glob("*.csv")
        if path.stem == stem
        or path.stem.startswith(stem)
        or stem.startswith(path.stem)
    ]
    if matches:
        return matches[0]

    return None


# ── core plotting function ────────────────────────────────────────────────
def visualize(results_path: Path) -> None:
    resolved_path = _resolve_results_path(results_path)
    if resolved_path is None:
        print(f"Not found: {results_path} — skipping.")
        return

    if resolved_path != results_path:
        print(f"Resolved {results_path} → {resolved_path}")

    df = pd.read_csv(resolved_path)

    # keep only successful runs
    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()

    if df.empty:
        print(f"No successful runs in {results_path.name} — skipping.")
        return

    # figure out which metrics are present
    has_auc = "roc_auc" in df.columns
    has_f1  = "f1_macro" in df.columns
    metrics = ["accuracy"] + (["roc_auc"] if has_auc else []) + (["f1_macro"] if has_f1 else [])

    title      = DATASET_TITLES.get(resolved_path.name, resolved_path.stem)
    stem       = resolved_path.stem
    out_prefix = RESULTS_DIR / stem

    print()
    print("=" * 70)
    print(f"DATASET: {title}  ({resolved_path.name})")
    print("=" * 70)

    # ── 1. overall summary ────────────────────────────────────────────────
    overall = (
        df.groupby("model")[metrics]
        .mean()
        .round(4)
        .sort_values("roc_auc" if has_auc else "accuracy", ascending=False)
    )
    print("\nOVERALL AVERAGE (across all variants and seeds):")
    print(overall.to_string())
    print()

    # ── 2. per-variant pivot tables ───────────────────────────────────────
    per_variant = (
        df.groupby(["dataset_variant", "model"])[metrics]
        .mean()
        .round(4)
        .reset_index()
    )

    # save detailed csv
    summary_path = RESULTS_DIR / f"{stem}_summary.csv"
    per_variant.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    # print pivot for each metric
    for metric in metrics:
        pivot = per_variant.pivot(
            index="dataset_variant", columns="model", values=metric
        )
        print(f"\n{metric.upper()} by variant and model:")
        print(pivot.to_string())

    if plt is None:
        print("matplotlib is not installed in this Python environment; skipping chart generation.")
        return

    # ── 3. accuracy chart ─────────────────────────────────────────────────
    _plot_metric(
        per_variant   = per_variant,
        df            = df,
        metric        = "accuracy",
        title         = f"Model comparison: {title}",
        ylabel        = "Test accuracy",
        out_path      = Path(f"{out_prefix}_accuracy.png"),
        ylim          = (0.0, 1.05),
    )

    # ── 4. ROC AUC chart ──────────────────────────────────────────────────
    if has_auc:
        _plot_metric(
            per_variant = per_variant,
            df          = df,
            metric      = "roc_auc",
            title       = f"Model comparison: {title} (ROC AUC)",
            ylabel      = "ROC AUC",
            out_path    = Path(f"{out_prefix}_roc_auc.png"),
            ylim        = (0.0, 1.05),
        )

    # ── 5. F1 chart ───────────────────────────────────────────────────────
    if has_f1:
        _plot_metric(
            per_variant = per_variant,
            df          = df,
            metric      = "f1_macro",
            title       = f"Model comparison: {title} (F1 Macro)",
            ylabel      = "F1 Macro",
            out_path    = Path(f"{out_prefix}_f1.png"),
            ylim        = (0.0, 1.05),
        )

    # ── 6. special: many_class — plot accuracy vs n_classes ───────────────
    if "n_classes" in df.columns:
        _plot_vs_n_classes(df, title, out_prefix)

    # ── 7. special: causal shortcut — plot accuracy drop ──────────────────
    if "accuracy_drop" in df.columns:
        _plot_accuracy_drop(df, title, out_prefix)


# ── helper: generic metric line plot ─────────────────────────────────────
def _plot_metric(
    per_variant : pd.DataFrame,
    df          : pd.DataFrame,
    metric      : str,
    title       : str,
    ylabel      : str,
    out_path    : Path,
    ylim        : tuple[float, float] = (0.0, 1.05),
) -> None:
    pivot        = per_variant.pivot(index="dataset_variant", columns="model", values=metric)
    variant_order = list(dict.fromkeys(df["dataset_variant"]))  # preserve run order

    plt.figure(figsize=(12, 6))
    for model_name in pivot.columns:
        y_values = [
            pivot.loc[v, model_name] if v in pivot.index else None
            for v in variant_order
        ]
        plt.plot(variant_order, y_values, marker="o", label=model_name)

    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Dataset variant")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved chart: {out_path}")


# ── helper: many_class — accuracy vs n_classes ────────────────────────────
def _plot_vs_n_classes(
    df          : pd.DataFrame,
    title       : str,
    out_prefix  : Path,
) -> None:
    """
    Special chart for many_class — x axis is number of classes,
    not dataset variant name. Makes the degradation trend much clearer.
    Includes a vertical dashed line at TabPFN v2's hard class limit (10).
    """
    agg = (
        df.groupby(["n_classes", "model"])[["accuracy", "f1_macro"]]
        .mean()
        .round(4)
        .reset_index()
    )

    for metric, ylabel in [("accuracy", "Test Accuracy"), ("f1_macro", "F1 Macro")]:
        plt.figure(figsize=(10, 6))
        for model_name, group in agg.groupby("model"):
            group = group.sort_values("n_classes")
            plt.plot(
                group["n_classes"],
                group[metric],
                marker="o",
                label=model_name,
            )

        # mark TabPFN v2 hard limit
        plt.axvline(
            x=10,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="TabPFN-v2 class limit (10)",
        )

        plt.xlabel("Number of classes")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {ylabel} vs Number of Classes")
        plt.ylim(0.0, 1.05)
        plt.xticks([2, 5, 10, 15, 20, 25])
        plt.legend()
        plt.tight_layout()
        out_path = Path(f"{out_prefix}_{metric}_vs_nclasses.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved chart: {out_path}")


# ── helper: causal shortcut — accuracy drop bar chart ────────────────────
def _plot_accuracy_drop(
    df          : pd.DataFrame,
    title       : str,
    out_prefix  : Path,
) -> None:
    """
    For causal shortcut datasets — shows how much each model's accuracy
    drops from test to ood_test. Bigger drop = more shortcut reliance.
    """
    agg = (
        df.groupby("model")["accuracy_drop"]
        .mean()
        .round(4)
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(9, 5))
    colors = [
        "tomato" if drop > 0.05 else "steelblue"
        for drop in agg["accuracy_drop"]
    ]
    plt.bar(agg["model"], agg["accuracy_drop"], color=colors)
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.axhline(
        y=0.05,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="5% drop threshold",
    )
    plt.xlabel("Model")
    plt.ylabel("Accuracy drop (test → ood_test)")
    plt.title(f"{title}: Shortcut Reliance (higher = worse)")
    plt.legend()
    plt.tight_layout()
    out_path = Path(f"{out_prefix}_accuracy_drop.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved chart: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to a specific results CSV (e.g. results/many_class_check_results.csv)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run visualizer on ALL csv files in the results/ directory",
    )
    args = parser.parse_args()

    if args.all:
        csv_files = sorted(RESULTS_DIR.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {RESULTS_DIR}/")
            return
        for csv_file in csv_files:
            visualize(csv_file)
    elif args.results:
        visualize(Path(args.results))
    else:
        # default: run on all known result files
        for filename in DATASET_TITLES:
            path = RESULTS_DIR / filename
            if path.exists():
                visualize(path)


if __name__ == "__main__":
    main()