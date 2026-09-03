import numpy as np
import pandas as pd
import pytest

from python.eval.metrics import (
    classification_metrics,
    naive_cancel_rate_baseline_predictions,
    recall_by_group,
    select_threshold_by_f1,
    select_threshold_by_macro_f1,
)


def test_classification_metrics_hand_computed():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    m = classification_metrics(y_true, y_pred)
    assert m == {
        "tp": 2, "fp": 1, "tn": 1, "fn": 1,
        "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "fpr": 0.5,
    }


def test_classification_metrics_all_negative_predictions_gives_zero_precision_recall():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["fpr"] == 0.0


def test_naive_baseline_thresholds_on_train_quantile_only():
    train = pd.DataFrame({"cancel_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
    test = pd.DataFrame({"cancel_rate": [0.85, 0.95]})
    pred, threshold = naive_cancel_rate_baseline_predictions(train, test, quantile=0.9)
    assert threshold == pytest.approx(0.91)
    assert list(pred) == [False, True]


def test_select_threshold_by_f1_finds_perfect_split_below_default_0_5():
    y_true = np.array([1, 1, 1, 0, 0])
    y_proba = np.array([0.30, 0.35, 0.40, 0.10, 0.20])
    threshold, metrics = select_threshold_by_f1(y_true, y_proba)
    assert threshold == pytest.approx(0.30)
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_select_threshold_by_f1_never_does_worse_than_default_0_5():
    y_true = np.array([1, 0, 1, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    threshold, metrics = select_threshold_by_f1(y_true, y_proba)
    default_metrics = classification_metrics(y_true, y_proba >= 0.5)
    assert metrics["f1"] >= default_metrics["f1"]


def test_select_threshold_by_macro_f1_does_not_zero_out_the_minority_group():
    # A large "layering" positive class (100 rows, all proba 1.0) and a
    # small "spoofing" positive class (3 rows, all proba 0.8), plus 30
    # false positives at proba 0.85. Pushing the threshold to 1.0 removes
    # every false positive at the cost of the entire spoofing group's
    # recall (0/3) — and still wins on pooled F1, because 100 layering
    # rows dominate the pooled numerator. Regression for exactly the
    # failure `select_threshold_by_f1` can produce under a minority class.
    y_true = np.array([1] * 100 + [1] * 3 + [0] * 30 + [0] * 20)
    y_proba = np.array([1.0] * 100 + [0.8] * 3 + [0.85] * 30 + [0.0] * 20)
    groups = pd.Series(["layering"] * 100 + ["spoofing"] * 3 + [None] * 30 + [None] * 20)

    pooled_threshold, pooled_metrics = select_threshold_by_f1(y_true, y_proba)
    assert pooled_threshold >= 0.9
    spoofing_mask = (y_true == 1) & (groups == "spoofing").to_numpy()
    assert np.all(y_proba[spoofing_mask] < pooled_threshold)  # pooled sacrifices spoofing

    macro_threshold, macro_metrics = select_threshold_by_macro_f1(y_true, y_proba, groups)
    assert np.all(y_proba[spoofing_mask] >= macro_threshold)  # macro keeps catching it
    assert macro_metrics["recall"] == 1.0


def test_recall_by_group_only_scores_positive_orders():
    test = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "label": [1, 1, 0, 1],
    })
    y_pred = np.array([1, 0, 1, 1])  # order 3 (label 0) predicted positive is ignored
    ground_truth = pd.DataFrame([
        {"order_ids": "1", "difficulty_tier": "easy"},
        {"order_ids": "2", "difficulty_tier": "hard"},
        {"order_ids": "4", "difficulty_tier": "hard"},
    ])
    recall = recall_by_group(test, y_pred, ground_truth, "difficulty_tier")
    assert recall["easy"] == 1.0
    assert recall["hard"] == 0.5  # order 2 missed, order 4 caught
