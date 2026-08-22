"""Builds a labeled training table from FeatureEngine output + ground truth.

`spoofwatch_features` (cpp/src/feature_dump.cpp) emits one feature-vector
row per order lifecycle event. For training, we want exactly one row per
order, taken at the moment it's submitted (the "NEW" event) — that's the
only point a real-time system could act on before knowing whether the
order gets cancelled or filled. Using a later snapshot would leak the
outcome we're trying to predict into the features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from python.injection.injector import load_ground_truth

FEATURE_COLUMNS = [
    "order_to_trade_ratio",
    "cancel_rate",
    "mean_lifetime_ns",
    "lifetime_stddev_ns",
    "layering_score_bid",
    "layering_score_ask",
    "cancel_burst_zscore",
    "size_vs_baseline_ratio",
]


def _manipulative_order_ids(ground_truth: pd.DataFrame) -> set[int]:
    ids: set[int] = set()
    for cell in ground_truth["order_ids"]:
        ids.update(int(x) for x in cell.split(";"))
    return ids


def build_training_table(features_csv: Path, ground_truth_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    ground_truth = load_ground_truth(ground_truth_csv)

    at_submission = features[features["event"] == "NEW"].copy()
    # An order_id should appear at most once as a NEW event; if the same
    # id somehow recurs (shouldn't, given LOBSTER's monotonic order_id
    # assignment plus injection.py's disjoint synthetic ranges), keep the
    # first submission.
    at_submission = at_submission.drop_duplicates(subset="order_id", keep="first")

    manipulative_ids = _manipulative_order_ids(ground_truth)
    at_submission["label"] = at_submission["order_id"].isin(manipulative_ids).astype(int)

    return at_submission.sort_values("time").reset_index(drop=True)


def time_based_split(table: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by time, not randomly — train on the earlier part of the day,
    evaluate on the later part, matching how the model would actually be
    deployed (never trained on data from the future relative to a live event).
    """
    split_idx = int(len(table) * (1 - test_frac))
    return table.iloc[:split_idx], table.iloc[split_idx:]
