"""
Summarizes recent failure case datasets (prior_distribution, periodic_boundary, context_length)
into clean accuracy + ROC AUC comparisons per model, averaged across seeds.

Prints summary tables and generates comparison charts for all dataset variants.

Run from the project root:
    python3 summarize_recent_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")

# Matches both standalone *_check_results.csv and main.py *_results.csv naming
TARGET_DATASETS = [
    "prior_distribution",
    "periodic_boundary",
    "context_length",
]


def find_csv_for_dataset(dataset_name: str) -> Path | None:
    """Finds existing CSV file for a dataset, preferring check_results if present."""
    candidates = [
        RESULTS_DIR / f"{dataset_name}_check_results.csv",
        RESULTS_DIR / f"{dataset_name}_results.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def process_dataset_results(dataset_name: str, csv_path: Path) -> pd.DataFrame | None:
    """Prints console summary and creates individual dataset plot."""
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "success"].copy()

    if df.empty:
        print(f"[{dataset_name}] No successful runs found in {csv_path.name}")
        return None

    # Available metric check (roc_auc might be missing/NaN in multi-class edge cases)
    metrics_to_group = ["accuracy", "f1_macro"]
    if "roc_auc" in df.columns and df["roc_auc"].notna().any():
        metrics_to_group.append("roc_auc")

    # ---------------------------------------------------------------
    # 1. Overall per-model summary for this dataset
    # ---------------------------------------------------------------
    overall = (
        df.groupby("model")[metrics_to_group]
        .mean()
        .round(4)
        .sort_values(metrics_to_group[0], ascending=False)
    )
    print("=" * 70)
    print(f"DATASET: {dataset_name.upper()} — OVERALL AVERAGE (across seeds)")
    print("=" * 70)
    print(overall.to_string())
    print()

    # ---------------------------------------------------------------
    # 2. Per-variant, per-model table
    # ---------------------------------------------------------------
    per_variant = (
        df.groupby(["dataset_variant", "model"])[metrics_to_group]
        .mean()
        .round(4)
        .reset_index()
    )

    pivot_acc = per_variant.pivot(
        index="dataset_variant", columns="model", values="accuracy"
    )

    print("-" * 70)
    print(f"Accuracy by variant for {dataset_name}")
    print("-" * 70)
    print(pivot_acc.to_string())
    print()

    # Save summary CSV
    summary_out = RESULTS_DIR / f"{dataset_name}_summary.csv"
    per_variant.to_csv(summary_out, index=False)
    print(f"Saved detailed table: {summary_out}")

    # ---------------------------------------------------------------
    # 3. Save individual ROC AUC / Accuracy chart for this dataset
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    variant_order = list(dict.fromkeys(df["dataset_variant"]))

    # Plot Accuracy
    for model_name in pivot_acc.columns:
        y_vals = [
            pivot_acc.loc[v, model_name] if v in pivot_acc.index else None
            for v in variant_order
        ]
        axes[0].plot(variant_order, y_vals, marker="o", label=model_name)

    axes[0].set_title(f"{dataset_name}: Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Plot ROC AUC if available
    if "roc_auc" in metrics_to_group:
        pivot_auc = per_variant.pivot(
            index="dataset_variant", columns="model", values="roc_auc"
        )
        for model_name in pivot_auc.columns:
            y_vals = [
                pivot_auc.loc[v, model_name] if v in pivot_auc.index else None
                for v in variant_order
            ]
            axes[1].plot(variant_order, y_vals, marker="s", label=model_name)
        axes[1].set_title(f"{dataset_name}: ROC AUC")
        axes[1].set_ylabel("ROC AUC")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()
    else:
        axes[1].text(
            0.5, 0.5, "ROC AUC Not Available", ha="center", va="center"
        )

    plt.tight_layout()
    chart_out = RESULTS_DIR / f"{dataset_name}_comparison.png"
    plt.savefig(chart_out, dpi=160)
    plt.close()
    print(f"Saved chart: {chart_out}\n")

    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    print("\n" + "#" * 70)
    print("      SUMMARIZING KATE'S RECENT FAILURE CASE RESULTS")
    print("#" * 70 + "\n")

    for dataset in TARGET_DATASETS:
        csv_path = find_csv_for_dataset(dataset)
        if csv_path is None:
            print(f"Skipping {dataset}: No result CSV found in {RESULTS_DIR}/")
            print(f"  Run: python -m datasets.{dataset} first.\n")
            continue

        df_processed = process_dataset_results(dataset, csv_path)
        if df_processed is not None:
            all_dfs.append(df_processed)

    # ---------------------------------------------------------------
    # 4. Generate grand overall comparison chart across all datasets
    # ---------------------------------------------------------------
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        overall_grand = (
            combined_df.groupby("model")[["accuracy", "f1_macro"]]
            .mean()
            .round(4)
            .sort_values("accuracy", ascending=False)
        )

        print("=" * 70)
        print("GRAND TOTAL AVERAGE ACROSS ALL NEW FAILURE CASES")
        print("=" * 70)
        print(overall_grand.to_string())
        print("=" * 70)


if __name__ == "__main__":
    main()