"""
Summarizes kate_datasets_results.csv into a clean accuracy + ROC AUC
comparison per model, averaged across seeds and dataset variants, plus a
per-variant ROC AUC chart (ROC AUC is more informative than accuracy alone
for spotting failure cases, since it's threshold-independent and more
sensitive to a model losing calibration/separability even when accuracy
looks fine - e.g. the imbalanced variants).

Run from the project root:
    python3 summarize_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("results/kate_datasets_results.csv")
OUT_CHART_PATH = Path("results/roc_auc_comparison.png")


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Not found: {RESULTS_PATH} - run main.py --dataset kate_datasets first.")
        return

    df = pd.read_csv(RESULTS_PATH)
    df = df[df["status"] == "success"].copy()

    # ---------------------------------------------------------------
    # 1. Overall per-model summary, averaged across every variant/seed
    # ---------------------------------------------------------------
    overall = (
        df.groupby("model")[["accuracy", "roc_auc", "f1_macro"]]
        .mean()
        .round(4)
        .sort_values("roc_auc", ascending=False)
    )
    print("=" * 70)
    print("OVERALL AVERAGE (across all variants and seeds)")
    print("=" * 70)
    print(overall.to_string())
    print()

    # ---------------------------------------------------------------
    # 2. Per-variant, per-model table (averaged across seeds) - the
    #    detailed numbers behind the accuracy chart you already have,
    #    but with ROC AUC added alongside.
    # ---------------------------------------------------------------
    per_variant = (
        df.groupby(["dataset_variant", "model"])[["accuracy", "roc_auc", "f1_macro"]]
        .mean()
        .round(4)
        .reset_index()
    )
    pivot_auc = per_variant.pivot(index="dataset_variant", columns="model", values="roc_auc")
    print("=" * 70)
    print("ROC AUC by dataset variant and model (averaged across seeds)")
    print("=" * 70)
    print(pivot_auc.to_string())
    print()

    per_variant.to_csv("results/accuracy_rocauc_summary.csv", index=False)
    print("Saved detailed table: results/accuracy_rocauc_summary.csv")

    # ---------------------------------------------------------------
    # 3. ROC AUC comparison chart, same layout as your existing accuracy
    #    chart, so it's a direct visual comparison.
    # ---------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    variant_order = list(dict.fromkeys(df["dataset_variant"]))  # first-seen order
    for model_name in pivot_auc.columns:
        y_values = [pivot_auc.loc[v, model_name] if v in pivot_auc.index else None for v in variant_order]
        plt.plot(variant_order, y_values, marker="o", label=model_name)

    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Dataset variant")
    plt.ylabel("ROC AUC")
    plt.title("Model comparison: kate_datasets (ROC AUC)")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHART_PATH, dpi=160)
    print(f"Saved chart: {OUT_CHART_PATH}")


if __name__ == "__main__":
    main()