import pandas as pd

from python.training.dataset import (
    attach_group_columns,
    build_training_table,
    time_based_split,
    time_based_split_train_val_test,
)


def _write(tmp_path, features_rows, ground_truth_rows):
    features_csv = tmp_path / "features.csv"
    ground_truth_csv = tmp_path / "ground_truth.csv"
    pd.DataFrame(features_rows).to_csv(features_csv, index=False)
    pd.DataFrame(ground_truth_rows).to_csv(ground_truth_csv, index=False)
    return features_csv, ground_truth_csv


def _feature_row(time, event, order_id, participant_id=1, **overrides):
    row = {
        "time": time, "event": event, "order_id": order_id,
        "participant_id": participant_id, "side": "BID",
        "order_to_trade_ratio": 1.0, "cancel_rate": 0.0,
        "mean_lifetime_ns": 0.0, "lifetime_stddev_ns": 0.0,
        "layering_score_bid": 1, "layering_score_ask": 0,
        "cancel_burst_zscore": 0.0, "size_vs_baseline_ratio": 1.0,
    }
    row.update(overrides)
    return row


def test_build_training_table_keeps_only_new_events_and_labels_correctly(tmp_path):
    features = [
        _feature_row(1.0, "NEW", 100),
        _feature_row(2.0, "DELETE", 100),
        _feature_row(3.0, "NEW", 101),
        _feature_row(4.0, "EXEC_VISIBLE", 101),
        _feature_row(5.0, "NEW", 999, participant_id=2000000),
        _feature_row(6.0, "DELETE", 999, participant_id=2000000),
    ]
    ground_truth = [{
        "pattern_id": 0, "pattern_type": "spoofing", "difficulty_tier": "easy",
        "participant_id": 2000000, "side": 1, "order_ids": "999",
        "start_time": 5.0, "end_time": 6.0,
    }]
    features_csv, ground_truth_csv = _write(tmp_path, features, ground_truth)

    table = build_training_table(features_csv, ground_truth_csv)

    assert set(table["order_id"]) == {100, 101, 999}
    assert (table["event"] == "NEW").all()
    labels = dict(zip(table["order_id"], table["label"]))
    assert labels == {100: 0, 101: 0, 999: 1}
    assert list(table["time"]) == sorted(table["time"])


def test_build_training_table_handles_multi_order_layering_labels(tmp_path):
    features = [
        _feature_row(1.0, "NEW", 1, participant_id=2000001),
        _feature_row(1.1, "NEW", 2, participant_id=2000001),
        _feature_row(1.2, "DELETE", 1, participant_id=2000001),
        _feature_row(1.3, "DELETE", 2, participant_id=2000001),
        _feature_row(2.0, "NEW", 500),
    ]
    ground_truth = [{
        "pattern_id": 0, "pattern_type": "layering", "difficulty_tier": "hard",
        "participant_id": 2000001, "side": -1, "order_ids": "1;2",
        "start_time": 1.0, "end_time": 1.3,
    }]
    features_csv, ground_truth_csv = _write(tmp_path, features, ground_truth)

    table = build_training_table(features_csv, ground_truth_csv)
    labels = dict(zip(table["order_id"], table["label"]))
    assert labels == {1: 1, 2: 1, 500: 0}


def test_attach_group_columns_only_labels_manipulative_orders():
    table = pd.DataFrame({"order_id": [1, 2, 3], "label": [1, 1, 0]})
    ground_truth = pd.DataFrame([
        {"order_ids": "1;2", "difficulty_tier": "hard", "pattern_type": "layering"},
    ])
    result = attach_group_columns(table, ground_truth)
    assert result["difficulty_tier"].tolist()[:2] == ["hard", "hard"]
    assert pd.isna(result["difficulty_tier"].iloc[2])
    assert result["pattern_type"].tolist()[:2] == ["layering", "layering"]
    assert pd.isna(result["pattern_type"].iloc[2])


def test_time_based_split_is_chronological_not_random():
    table = pd.DataFrame({"time": range(10), "label": [0] * 10})
    train, test = time_based_split(table, test_frac=0.3)
    assert len(train) == 7 and len(test) == 3
    assert train["time"].max() < test["time"].min()


def test_time_based_split_train_val_test_is_chronological_and_disjoint():
    table = pd.DataFrame({"time": range(10), "label": [0] * 10})
    train, val, test = time_based_split_train_val_test(table, val_frac=0.2, test_frac=0.3)
    assert len(train) == 5 and len(val) == 2 and len(test) == 3
    assert train["time"].max() < val["time"].min()
    assert val["time"].max() < test["time"].min()
