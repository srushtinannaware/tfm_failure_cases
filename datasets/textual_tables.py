"""
Textual Tables Dataset:
Tests whether a model extracts SEMANTIC signal from a free-text column,
or only treats it as an opaque categorical ID.

Each row's label depends on the net sentiment of adjectives embedded in
a short templated sentence. Sentences are template + randomly sampled
words, so almost every exact string in the test set is novel (never
seen verbatim in train) — a model that label-encodes/hashes the text as
a categorical ID cannot generalize; only genuine token-level parsing
(bag-of-words, embeddings) can.

Why this is plausibly the "publishable" one: per Prior Labs' own model
docs, "native text features" are explicitly NOT available in the
open-source TabPFN-3 package (only in the hosted/Plus tier), and TabICL
has no text-handling path at all — both default to categorical/ordinal
encoding of raw strings. CatBoost DOES have native text-feature support
(bag-of-words / embeddings), so this dataset doubles as a control: if
CatBoost solves it while the ICL models collapse to chance, that
isolates a REPRESENTATION gap (fixable by feeding embeddings) rather
than a capacity limit — a more precise and more publishable claim than
"transformers are bad at text-ish tables."

TODO(kate): this returns raw text in an object-dtype column — your
evaluate_classifier / models.py wrapper needs to either (a) pass a
text/categorical feature index through to CatBoost's Pool so it uses
its native text handling, or (b) you'll need a separate code path that
skips CatBoost's advantage entirely and compares everyone on raw
label-encoded strings. Right now these are two different experiments
that answer different questions — decide which one you want before
running the full sweep, since results will not be comparable across
model families otherwise.

Variants:
  - text_sentiment_clean: single text column, label = net-positive
    word count > 0, no numeric noise.
  - text_sentiment_noisy: same signal, +20 pure numeric noise columns
    (tests whether text signal gets crowded out alongside irrelevant
    numeric columns).
  - text_sentiment_ambiguous: adjectives can be negated ("not great"),
    requiring 2-token compositional parsing rather than single-keyword
    spotting.
  - text_sentiment_embedded_control: same underlying task as *_noisy,
    but the text column is REPLACED by one clean numeric column
    exposing the true sentiment signal directly — a stand-in for "given
    a good text embedding instead of raw text". This is the ablation
    that turns a capability gap into a representation-gap claim: if
    models jump from chance on *_noisy to strong on this variant, the
    text itself (not reasoning) was the bottleneck.

Usage:
    python main.py --dataset textual_tables --models catboost realmlp tabpfn_v2 tabpfn_v3 tabicl_v2
    python -m datasets.textual_tables
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
import sys

import numpy as np

N_TRAIN = 1000
N_TEST = 200
MASTER_SEED_OFFSET = 900
LABEL_FLIP_PROB = 0.02

POSITIVE_WORDS = ["great", "excellent", "wonderful", "solid", "impressive", "reliable"]
NEGATIVE_WORDS = ["terrible", "poor", "disappointing", "flawed", "weak", "unreliable"]
TEMPLATES = [
    "The product was {adj} overall.",
    "Customer service felt {adj} to us.",
    "Build quality seemed {adj} for the price.",
    "Delivery experience was {adj} this time.",
]


def _make_sentiment_text(n: int, rng, negatable: bool = False) -> tuple[np.ndarray, np.ndarray]:
    templates = rng.integers(0, len(TEMPLATES), size=n)
    is_positive = rng.integers(0, 2, size=n).astype(bool)
    negated = rng.random(n) < 0.3 if negatable else np.zeros(n, dtype=bool)

    texts = np.empty(n, dtype=object)
    net_positive = np.zeros(n, dtype=bool)

    for i in range(n):
        word = rng.choice(POSITIVE_WORDS) if is_positive[i] else rng.choice(NEGATIVE_WORDS)
        adj = f"not {word}" if negated[i] else word
        texts[i] = TEMPLATES[templates[i]].format(adj=adj)
        # negation flips effective sentiment
        net_positive[i] = is_positive[i] ^ negated[i]

    y = net_positive.astype(int)
    flips = rng.random(n) < LABEL_FLIP_PROB
    y = np.where(flips, 1 - y, y)
    return texts, y


def get_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed + MASTER_SEED_OFFSET)
    n_total = N_TRAIN + N_TEST
    datasets = {}

    texts, y = _make_sentiment_text(n_total, rng, negatable=False)
    datasets["text_sentiment_clean"] = (texts.reshape(-1, 1), y)

    texts, y = _make_sentiment_text(n_total, rng, negatable=False)
    noise = rng.normal(0, 1, size=(n_total, 20))
    X_noisy = np.hstack([texts.reshape(-1, 1), noise])
    datasets["text_sentiment_noisy"] = (X_noisy, y)

    texts, y = _make_sentiment_text(n_total, rng, negatable=True)
    datasets["text_sentiment_ambiguous"] = (texts.reshape(-1, 1), y)

    texts, y_ctrl = _make_sentiment_text(n_total, rng, negatable=False)
    signal_col = (y_ctrl.astype(float) * 2 - 1) + rng.normal(0, 0.1, n_total)
    noise_ctrl = rng.normal(0, 1, size=(n_total, 20))
    X_ctrl = np.hstack([signal_col.reshape(-1, 1), noise_ctrl])
    datasets["text_sentiment_embedded_control"] = (X_ctrl, y_ctrl)

    return datasets


def run_textual_tables_check(model_names: list[str], seeds: list[int] = (0, 1, 2)) -> list[dict]:
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
    csv_path = results_dir / "textual_tables_results.csv"

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
    print("Running Textual Tables Benchmark")
    print("=" * 70)
    run_textual_tables_check(model_names=models_to_run)
