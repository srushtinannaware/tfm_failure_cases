"""Create poster-ready GSE40279 age-regression plots and a summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "tabpfn_v2_5": "TabPFN-2.5",
    "tabpfn_v3": "TabPFN-3",
    "tabicl_v2": "TabICL v2",
    "xgboost": "XGBoost",
    "ridge": "Ridge",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/gse40279_age_regression.csv")
    parser.add_argument("--outdir", default="results/gse40279_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    data = data[data.status == "ok"].copy()
    keys = ["model", "n_features", "repeat", "fold"]
    data = data.drop_duplicates(keys, keep="last")
    for column in ["r2", "mae", "rmse", "fit_seconds", "predict_seconds"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    summary = (
        data.groupby(["model", "n_features"])
        .agg(
            folds=("r2", "count"), r2_mean=("r2", "mean"), r2_std=("r2", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            predict_seconds_mean=("predict_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "gse40279_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in summary.model.unique():
        model = summary[summary.model == model_key].sort_values("n_features")
        ax.errorbar(
            model.n_features, model.r2_mean, yerr=model.r2_std.fillna(0),
            marker="o", linewidth=2, capsize=4, label=LABELS.get(model_key, model_key),
        )
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_xticks(sorted(summary.n_features.unique()))
    ax.set_xlabel("CpG features selected within each training fold")
    ax.set_ylabel("Cross-validation R² (mean ± SD across folds)")
    ax.set_title("Human blood DNA methylation age regression — GSE40279")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "gse40279_age_r2_vs_features.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {outdir / 'gse40279_summary.csv'}")
    print(f"Wrote {outdir / 'gse40279_age_r2_vs_features.png'}")


if __name__ == "__main__":
    main()
