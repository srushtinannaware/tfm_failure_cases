"""Leakage-safe cross-validation for GSE40279 age regression.

Run neural foundation models and XGBoost in separate processes on macOS.  The
same deterministic folds are reconstructed in every invocation.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from models import MODEL_KEYS, get_model, get_tabicl_device, get_tabpfn_device


ALL_MODELS = MODEL_KEYS + ["ridge"]
FIELDS = [
    "dataset", "model", "n_features", "selection", "repeat", "fold",
    "n_train", "n_test", "status", "r2", "mae", "rmse", "fit_seconds",
    "predict_seconds", "selected_feature_hash", "error",
]


def fold_variances(X: np.ndarray, train_idx: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
    variances = np.empty(X.shape[1], dtype=np.float64)
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        block = np.asarray(X[train_idx, start:stop], dtype=np.float64)
        variances[start:stop] = np.nanvar(block, axis=0)
    variances[~np.isfinite(variances)] = -np.inf
    return variances


def fold_age_correlations(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, chunk_size: int = 10_000
) -> np.ndarray:
    scores = np.empty(X.shape[1], dtype=np.float64)
    target = y[train_idx].astype(np.float64)
    target = target - target.mean()
    target_norm = np.sqrt(np.sum(target**2))
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        block = np.asarray(X[np.ix_(train_idx, np.arange(start, stop))], dtype=np.float64)
        means = np.nanmean(block, axis=0)
        missing = np.where(np.isnan(block))
        if missing[0].size:
            block[missing] = means[missing[1]]
        block -= means
        denominator = np.sqrt(np.sum(block**2, axis=0)) * target_norm
        scores[start:stop] = np.divide(
            np.abs(target @ block), denominator,
            out=np.full(stop - start, -np.inf), where=denominator > 0,
        )
    scores[~np.isfinite(scores)] = -np.inf
    return scores


def select_features(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, n_features: int, method: str
) -> np.ndarray:
    if method == "variance":
        scores = fold_variances(X, train_idx)
    elif method == "correlation":
        scores = fold_age_correlations(X, y, train_idx)
    else:
        raise ValueError(f"Unknown selection method: {method}")
    usable = int(np.isfinite(scores).sum())
    if usable < n_features:
        raise ValueError(f"Only {usable} usable CpGs; requested {n_features}")
    chosen = np.argpartition(scores, -n_features)[-n_features:]
    # Stable ordering makes every model receive exactly the same matrix.
    return chosen[np.argsort(scores[chosen])[::-1]]


def make_model(model_key: str):
    if model_key == "ridge":
        return Ridge(alpha=1.0)
    return get_model(model_key, "regression")


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def existing_keys(path: Path) -> set[tuple[str, int, str, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (
                row["model"], int(row["n_features"]), row["selection"],
                int(row["repeat"]), int(row["fold"]),
            )
            for row in csv.DictReader(handle)
            if row["status"] == "ok"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/gse40279")
    parser.add_argument("--output", default="results/gse40279_age_regression.csv")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, required=True)
    parser.add_argument("--feature-counts", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--selection", choices=["correlation", "variance"], default="correlation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "methylation.npy", mmap_mode="r")
    y = np.load(data_dir / "ages.npy").astype(np.float64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch: {X.shape[0]} vs {y.shape[0]}")

    # A fixed random subset is used only if the prepared block exceeds the cap.
    subset_rng = np.random.default_rng(args.seed)
    indices = np.arange(len(y))
    if len(indices) > args.max_samples:
        indices = np.sort(subset_rng.choice(indices, args.max_samples, replace=False))
        X = X[indices]
        y = y[indices]

    output = Path(args.output)
    done = existing_keys(output)
    device_bits = []
    if any(model in args.models for model in ("tabpfn_v2_5", "tabpfn_v3")):
        device_bits.append(f"TabPFN device: {get_tabpfn_device()}")
    if "tabicl_v2" in args.models:
        device_bits.append(f"TabICL device: {get_tabicl_device()}")
    print(" | ".join(device_bits) if device_bits else "Classical models only")

    for repeat in range(args.repeats):
        if args.folds == 1:
            splitter = ShuffleSplit(
                n_splits=1, test_size=args.test_size, random_state=args.seed + repeat
            )
        elif args.folds >= 2:
            splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + repeat)
        else:
            raise ValueError("--folds must be at least 1")
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)))):
            # Rank all CpGs once for this fold using training people only.
            ranking = select_features(
                X, y, train_idx, max(args.feature_counts), args.selection
            )
            for n_features in args.feature_counts:
                chosen = ranking[:n_features]
                feature_hash = __import__("hashlib").sha256(chosen.tobytes()).hexdigest()[:16]
                # np.ix_ reads only the selected rows/columns from the
                # memory-mapped 300+ MB matrix instead of copying all CpGs.
                X_train = np.asarray(X[np.ix_(train_idx, chosen)], dtype=np.float32)
                X_test = np.asarray(X[np.ix_(test_idx, chosen)], dtype=np.float32)
                imputer = SimpleImputer(strategy="median")
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)

                for model_key in args.models:
                    key = (model_key, n_features, f"train_{args.selection}", repeat, fold)
                    if key in done:
                        print(f"Skipping completed {key}")
                        continue
                    row = {
                        "dataset": "GSE40279", "model": model_key,
                        "n_features": n_features, "selection": f"train_{args.selection}",
                        "repeat": repeat, "fold": fold, "n_train": len(train_idx),
                        "n_test": len(test_idx), "status": "ok", "r2": "",
                        "mae": "", "rmse": "", "fit_seconds": "",
                        "predict_seconds": "", "selected_feature_hash": feature_hash,
                        "error": "",
                    }
                    label = f"features={n_features:>3} repeat={repeat} fold={fold} model={model_key}"
                    try:
                        # Scaling is useful for Ridge and harmlessly explicit;
                        # TFMs and XGBoost receive raw beta values.
                        if model_key == "ridge":
                            scaler = StandardScaler()
                            fit_X = scaler.fit_transform(X_train)
                            pred_X = scaler.transform(X_test)
                        else:
                            fit_X, pred_X = X_train, X_test
                        model = make_model(model_key)
                        start = time.perf_counter()
                        model.fit(fit_X, y[train_idx])
                        row["fit_seconds"] = time.perf_counter() - start
                        start = time.perf_counter()
                        predictions = np.asarray(model.predict(pred_X)).reshape(-1)
                        row["predict_seconds"] = time.perf_counter() - start
                        row["r2"] = r2_score(y[test_idx], predictions)
                        row["mae"] = mean_absolute_error(y[test_idx], predictions)
                        row["rmse"] = np.sqrt(mean_squared_error(y[test_idx], predictions))
                        print(f"{label} OK r2={row['r2']:.4f} mae={row['mae']:.2f}")
                    except Exception as error:  # continue and preserve failures
                        row["status"] = "error"
                        row["error"] = f"{type(error).__name__}: {error}"
                        print(f"{label} ERROR {row['error']}")
                        traceback.print_exc()
                    append_row(output, row)

    print(f"Done. Results written to {output}")


if __name__ == "__main__":
    main()
