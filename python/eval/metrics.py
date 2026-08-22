"""Precision/recall/F1/FPR evaluation, plus the naive baseline detector
that any real model needs to beat (docs/PHASES.md Phase 4 exit criterion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def naive_cancel_rate_baseline_predictions(
    train: pd.DataFrame, test: pd.DataFrame, quantile: float = 0.99,
) -> tuple[np.ndarray, float]:
    """Flags an order as manipulative if this participant's cancel_rate
    (as of that order's submission) exceeds a threshold fit on the
    training split alone — the simplest rule a naive surveillance system
    might ship with. Any real model needs to beat this.
    """
    threshold = float(train["cancel_rate"].quantile(quantile))
    predictions = (test["cancel_rate"] > threshold).to_numpy()
    return predictions, threshold


def recall_by_group(
    test: pd.DataFrame, y_pred: np.ndarray, ground_truth: pd.DataFrame, group_col: str,
) -> pd.Series:
    """Recall broken out by a ground-truth grouping column (difficulty_tier
    or pattern_type) — an aggregate recall number hides whether a model is
    only catching the easy cases.
    """
    order_to_group: dict[int, str] = {}
    for _, row in ground_truth.iterrows():
        for oid in row["order_ids"].split(";"):
            order_to_group[int(oid)] = row[group_col]

    positives = test[test["label"] == 1].copy()
    positives["_pred"] = y_pred[test["label"].to_numpy() == 1]
    positives["_group"] = positives["order_id"].map(order_to_group)

    return positives.groupby("_group")["_pred"].mean().rename("recall")
