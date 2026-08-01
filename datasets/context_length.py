"""
Feature Attention Dilution Dataset:
Tests transformer in-context attention bottlenecks under increasing feature dimensions.

Tree models perform greedy split selection to naturally ignore uninformative 
features. Transformers pass all feature tokens into self-attention layers,
diluting attention weights over noisy dimensions as feature count grows.

Variants:
  - attention_clean_base:       5 clean signal features (0 noise features).
  - attention_dilution_50feat:  5 clean signal features + 45 noise features.
  - attention_dilution_150feat: 5 clean signal features + 145 noise features.

Usage 1 (Standard main.py harness):
    python main.py --dataset context_length --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2

Usage 2 (Standalone module check matching deep_causal_chain.py pattern):
    python -m datasets.context_length
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

N_TRAIN = 1000
N_TEST = 200
N_SIGNAL_FEATURES = 5
MASTER_SEED_OFFSET = 500


def _generate_signal_and_labels(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X_signal = rng.normal(0, 1, size=(n, N_SIGNAL_FEATURES))
    
    # WEAKER signal — add noise directly to the score
    score = (
        X_signal[:, 0] * X_signal[:, 1]    # interaction term
        + X_signal[:, 2]                     # linear term
        - X_signal[:, 3] * X_signal[:, 4]   # interaction term
        + rng.normal(0, 1.5, n)              # ← add label noise here
    )                                         # makes signal weaker
    
    y = (score > 0).astype(int)
    return X_signal, y

def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets = {}

    X_signal, y = _generate_signal_and_labels(n_total, rng)

    # Variant 1: Clean Base (unchanged)
    datasets["attention_clean_base"] = (X_signal.copy(), y)

    # Variant 2: 50 features — but noise is CORRELATED with signal
    # This is much harder for TFMs because correlated noise looks
    # like real features in the attention matrix
    noise_correlated_45 = X_signal[:, :1] * 0.3 + rng.normal(0, 1, (n_total, 45))
    X_50_corr = np.hstack([X_signal, noise_correlated_45])
    datasets["attention_dilution_50feat_correlated"] = (X_50_corr, y)

    # Variant 3: 150 features — correlated noise
    noise_correlated_145 = X_signal[:, :1] * 0.3 + rng.normal(0, 1, (n_total, 145))
    X_150_corr = np.hstack([X_signal, noise_correlated_145])
    datasets["attention_dilution_150feat_correlated"] = (X_150_corr, y)

    # Variant 4: 500 features — this should break TFMs completely
    # TabPFN has a hard feature limit — beyond it performance collapses
    noise_500 = X_signal[:, :1] * 0.3 + rng.normal(0, 1, (n_total, 495))
    X_500 = np.hstack([X_signal, noise_500])
    datasets["attention_dilution_500feat"] = (X_500, y)

    # Variant 5: Few-shot — only 50 training samples
    # TFMs struggle with tiny context, CatBoost overfits but differently
    datasets["attention_fewshot_50train"] = (X_signal.copy(), y)
    # NOTE: in run_check below, use only first 50 rows for training
    # TabPFN v2 hard limit: 100 features
    # TabPFN v3 extended this but still degrades sharply above ~500
    # Test exactly at and beyond the limit:

    FEATURE_COUNTS = [5, 50, 100, 200, 500, 1000]

    for n_feat in FEATURE_COUNTS:
        noise = rng.normal(0, 1, (n_total, n_feat - N_SIGNAL_FEATURES))
        X = np.hstack([X_signal, noise])
        datasets[f"attention_dilution_{n_feat}feat"] = (X, y)

    return datasets


def run_context_length_check(
    model_names: list[str],
    seeds: list[int] = (0, 1, 2),
) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows: list[dict] = []

    for seed in seeds:
        dataset_variants = get_datasets(seed)

        for variant_name, (X, y) in dataset_variants.items():
            X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
            X_test, y_test = X[N_TRAIN:], y[N_TRAIN:]

            for model_name, factory in model_factories.items():
                print(f"[{variant_name}] {model_name} (seed={seed})...")

                base_row = {
                    "dataset_variant": variant_name,
                    "model": model_name,
                    "seed": seed,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "train_samples": int(X_train.shape[0]),
                    "test_samples": int(X_test.shape[0]),
                }

                try:
                    metrics = evaluate_classifier(
                        model=factory(),
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                    )

                    row = {**base_row, **metrics, "status": "success", "error": ""}
                    print(
                        f"  Accuracy={metrics['accuracy']:.4f} | "
                        f"F1={metrics['f1_macro']:.4f} | "
                        f"Log loss={metrics['log_loss']:.4f} | "
                        f"Time={metrics['total_seconds']:.2f}s"
                    )

                except Exception as error:
                    row = {**base_row, "status": "failed", "error": str(error)}
                    print(f"  Failed: {error}")

                rows.append(row)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "context_length_check_results.csv"

    all_columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in all_columns:
                all_columns.append(column)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {csv_path}")
    return rows


if __name__ == "__main__":
    models_to_run = ["catboost", "realmlp", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]

    print("=" * 70)
    print("Running Context Length & Feature Attention Dilution Benchmark")
    print("=" * 70)
    run_context_length_check(model_names=models_to_run)