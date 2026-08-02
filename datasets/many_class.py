from __future__ import annotations
import csv as csv_module
from pathlib import Path
import sys
import numpy as np

N_TOTAL = 5000
N_FEATURES = 10
# extended class counts — TabPFN v2 crashes at 15, v3 degrades beyond 20
CLASS_COUNTS = [2, 5, 10, 15, 20, 25, 30]
MASTER_SEED_OFFSET = 600


def _generate_voronoi(n, n_classes, seed):
    """Original clean Voronoi — well-separated clusters.
    Used as baseline to show models CAN handle many classes
    when clusters are clean. TabPFN v2 still crashes at 15."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(n_classes, N_FEATURES))
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * 6.0

    samples_per_class = n // n_classes
    remainder = n % n_classes
    X_list, y_list = [], []
    for c in range(n_classes):
        count = samples_per_class + (1 if c < remainder else 0)
        X_c = rng.normal(loc=centroids[c], scale=0.8, size=(count, N_FEATURES))
        X_list.append(X_c)
        y_list.append(np.full(count, c, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _generate_overlapping(n, n_classes, seed):
    """
    Overlapping clusters — centroids closer together, larger spread.
    Classes overlap significantly making boundaries ambiguous.

    Why TFMs fail specifically:
    - TFM prior trained on well-separated synthetic datasets
    - Overlapping classes = out-of-distribution for the prior
    - TFMs try to find clean boundaries that don't exist
    - CatBoost uses probabilistic splits — handles overlap naturally
      by learning soft decision boundaries via gradient boosting

    scale=1.5 (was 0.8) makes clusters overlap substantially.
    centroid_scale=3.0 (was 6.0) pulls centroids closer together.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(n_classes, N_FEATURES))
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * 3.0

    samples_per_class = n // n_classes
    remainder = n % n_classes
    X_list, y_list = [], []
    for c in range(n_classes):
        count = samples_per_class + (1 if c < remainder else 0)
        # scale=1.5 causes heavy overlap between nearby centroids
        X_c = rng.normal(loc=centroids[c], scale=1.5, size=(count, N_FEATURES))
        X_list.append(X_c)
        y_list.append(np.full(count, c, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _generate_imbalanced(n, n_classes, seed):
    """
    Imbalanced classes — some classes have 10x more samples than others.
    Class 0 gets 50% of samples, rest share the remaining 50%.

    Why TFMs fail specifically:
    - TFM prior was trained on balanced synthetic datasets
    - Severe imbalance = out-of-distribution for the prior
    - TFM attention gives equal weight to all context rows
      so rare classes get proportionally less attention signal
    - CatBoost handles imbalance via class weights in boosting
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(n_classes, N_FEATURES))
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * 6.0

    # class 0 gets 50%, rest share 50%
    n_majority = n // 2
    n_minority_each = (n - n_majority) // (n_classes - 1)
    remainder = n - n_majority - n_minority_each * (n_classes - 1)

    X_list, y_list = [], []
    for c in range(n_classes):
        if c == 0:
            count = n_majority
        elif c == n_classes - 1:
            count = n_minority_each + remainder
        else:
            count = n_minority_each
        X_c = rng.normal(loc=centroids[c], scale=0.8, size=(count, N_FEATURES))
        X_list.append(X_c)
        y_list.append(np.full(count, c, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def get_datasets(seed):
    datasets = {}

    for n_classes in CLASS_COUNTS:
        # variant 1: original clean Voronoi (baseline)
        X, y = _generate_voronoi(
            N_TOTAL, n_classes,
            seed + MASTER_SEED_OFFSET + n_classes
        )
        datasets[f"many_class_{n_classes:02d}"] = (X, y)

        # variant 2: overlapping clusters
        # only run for class counts where TabPFN v2 can still run (<=10)
        # and for higher counts to show progressive failure
        X, y = _generate_overlapping(
            N_TOTAL, n_classes,
            seed + MASTER_SEED_OFFSET + n_classes + 100
        )
        datasets[f"many_class_overlap_{n_classes:02d}"] = (X, y)

        # variant 3: imbalanced classes
        X, y = _generate_imbalanced(
            N_TOTAL, n_classes,
            seed + MASTER_SEED_OFFSET + n_classes + 200
        )
        datasets[f"many_class_imbal_{n_classes:02d}"] = (X, y)

    return datasets


def run_many_class_check(model_names, seeds=(0, 1), train_ratio=0.8):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics import evaluate_classifier
    from models import get_model

    model_factories = get_model(model_names)
    rows = []
    n_train = int(N_TOTAL * train_ratio)
    n_test = N_TOTAL - n_train

    for seed in seeds:
        dataset_variants = get_datasets(seed)
        for variant_name, (X, y) in dataset_variants.items():
            # extract n_classes from variant name
            n_classes = int(variant_name.split("_")[-1])
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]

            for model_name, factory in model_factories.items():
                print(f"[{variant_name}] {model_name} (seed={seed})...")
                base_row = {
                    "dataset_variant" : variant_name,
                    "n_classes"       : n_classes,
                    "model"           : model_name,
                    "seed"            : seed,
                    "n_train"         : n_train,
                    "n_test"          : n_test,
                    "n_features"      : N_FEATURES,
                }
                try:
                    metrics = evaluate_classifier(
                        model   = factory(),
                        X_train = X_train,
                        X_test  = X_test,
                        y_train = y_train,
                        y_test  = y_test,
                    )
                    row = {**base_row, **metrics, "status": "success", "error": ""}
                    print(
                        f"  Accuracy={metrics['accuracy']:.4f} | "
                        f"F1={metrics['f1_macro']:.4f}"
                    )
                except Exception as error:
                    error_str = str(error)
                    status = (
                        "architectural_limit"
                        if "exceeds the maximum number of classes" in error_str
                        else "failed"
                    )
                    row = {**base_row, "status": status, "error": error_str}
                    print(f"  {status.upper()}: {error}")
                rows.append(row)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "many_class_check_results.csv"
    all_columns = []
    for row in rows:
        for col in row:
            if col not in all_columns:
                all_columns.append(col)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {csv_path}")
    return rows


if __name__ == "__main__":
    run_many_class_check(
        ["catboost", "tabpfn_v2", "tabpfn_v3", "tabicl_v2"]
    )