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


def select_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict]:
    """Picks the probability threshold that maximizes F1 on the given
    (labeled) split, instead of using LightGBM's default 0.5 — a poor
    operating point under this dataset's class imbalance (precision/recall/
    F1 at 0.5 are dominated by wherever the sigmoid happens to sit, not by
    the actual precision/recall tradeoff available).

    Only the scores that actually appear as candidate cutoffs change the
    resulting predictions, so every unique predicted probability (plus 0.0,
    to allow "flag everything") is tried exhaustively rather than sweeping
    an arbitrary grid.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_proba = np.asarray(y_proba)

    candidates = np.unique(np.concatenate([[0.0], y_proba]))
    best_threshold = 0.5
    best_metrics = classification_metrics(y_true, y_proba >= best_threshold)
    for threshold in candidates:
        m = classification_metrics(y_true, y_proba >= threshold)
        if m["f1"] > best_metrics["f1"]:
            best_threshold, best_metrics = float(threshold), m

    return best_threshold, best_metrics


def select_threshold_by_macro_f1(
    y_true: np.ndarray, y_proba: np.ndarray, groups: pd.Series,
) -> tuple[float, dict]:
    """Like `select_threshold_by_f1`, but instead of maximizing one pooled
    F1, maximizes the AVERAGE of per-group F1 (each group's own recall
    combined with the overall precision at that threshold) — `groups` is a
    label per row (e.g. pattern_type), only meaningful for positives.

    Plain pooled F1 can be maximized by writing off an entire minority
    group: pooling multiple symbols' injected patterns exposed exactly
    this — layering patterns are multi-order and vastly outnumber
    single-order spoofing patterns in raw row count, so a threshold that
    pushes spoofing recall to 0.0 to squeeze out a marginal precision gain
    on layering can still win on pooled F1, even though it's a worse
    detector overall. Averaging per-group F1 means a group that's been
    zeroed out drags the average down hard enough that it can't win.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_proba = np.asarray(y_proba)
    groups = pd.Series(groups).reset_index(drop=True)
    positive_groups = groups[y_true]
    unique_groups = [g for g in positive_groups.unique() if pd.notna(g)]

    def macro_f1_at(threshold: float) -> float:
        m = classification_metrics(y_true, y_proba >= threshold)
        precision = m["precision"]
        f1s = []
        for g in unique_groups:
            mask = y_true & (groups == g).to_numpy()
            recall_g = float(np.mean(y_proba[mask] >= threshold)) if mask.any() else 0.0
            f1s.append(
                2 * precision * recall_g / (precision + recall_g) if (precision + recall_g) else 0.0
            )
        return float(np.mean(f1s)) if f1s else m["f1"]

    candidates = np.unique(np.concatenate([[0.0], y_proba]))
    best_threshold = 0.5
    best_score = macro_f1_at(best_threshold)
    for threshold in candidates:
        score = macro_f1_at(threshold)
        if score > best_score:
            best_threshold, best_score = float(threshold), score

    return best_threshold, classification_metrics(y_true, y_proba >= best_threshold)


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
