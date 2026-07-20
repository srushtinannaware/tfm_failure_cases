"""Shared classification metrics.

Calculate metrics for classification models:
    accuracy,
    balanced accuracy,
    precision, recall,
    F1 score,
    Matthews correlation coefficient (MCC),
    log loss,
    ROC,
    AUC, 
    The module also measures the time taken for model fitting and prediction.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit and evaluate one classification model."""

    # ---------------------------- Training----------------------------
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    # ----------------------------Prediction----------------------------
    
    prediction_start = time.perf_counter()

    predictions = np.asarray(
        model.predict(X_test)
    ).reshape(-1)

    prediction_seconds = (
        time.perf_counter() - prediction_start
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "fit_seconds": float(
            fit_seconds
        ),

        "prediction_seconds": float(
            prediction_seconds
        ),

        "total_seconds": float(
            fit_seconds + prediction_seconds
        ),
    }

    # ---------------------------- Probability-based metrics----------------------------
    try:

        probabilities = np.asarray(
            model.predict_proba(X_test)
        )

        metrics["log_loss"] = float(
            log_loss(
                y_test,
                probabilities,
                labels=np.unique(y_train),
            )
        )
        n_classes = len(np.unique(y_train))
        if n_classes == 2:

            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )
            )
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )

    except (
        AttributeError,
        NotImplementedError,
        ValueError,
    ):

        metrics["log_loss"] = np.nan
        metrics["roc_auc"] = np.nan

    return metrics