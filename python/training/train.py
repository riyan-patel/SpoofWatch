"""Phase 4: trains a LightGBM spoofing/layering classifier on FeatureEngine
output, evaluates on a time-based held-out split, and compares against the
naive cancel-rate-threshold baseline.

Usage:
    python -m python.training.train \\
        data/synthetic/AAPL_2012-06-21/features.csv \\
        data/synthetic/AAPL_2012-06-21/ground_truth.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import average_precision_score

from python.eval.metrics import (
    classification_metrics,
    naive_cancel_rate_baseline_predictions,
    recall_by_group,
)
from python.injection.injector import load_ground_truth
from python.training.dataset import FEATURE_COLUMNS, build_training_table, time_based_split
from python.training.export_model import export_lightgbm_model


def _print_metrics(name: str, m: dict) -> None:
    print(
        f"{name:>28s}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
        f"f1={m['f1']:.3f}  fpr={m['fpr']:.4f}  "
        f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--baseline-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="if set, writes model.bin (for the C++ branchless evaluator) and "
             "reference_predictions.csv (test-set features + Python-computed "
             "probability, for cross-validating the C++ side) into this directory",
    )
    args = parser.parse_args()

    table = build_training_table(args.features_csv, args.ground_truth_csv)
    train, test = time_based_split(table, test_frac=args.test_frac)
    ground_truth = load_ground_truth(args.ground_truth_csv)

    print(
        f"{len(table)} orders total ({table['label'].sum()} manipulative) — "
        f"{len(train)} train / {len(test)} test (time-based split)"
    )
    print(f"train positives={train['label'].sum()}  test positives={test['label'].sum()}")

    baseline_pred, threshold = naive_cancel_rate_baseline_predictions(
        train, test, quantile=args.baseline_quantile
    )
    baseline_metrics = classification_metrics(test["label"].to_numpy(), baseline_pred)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=args.seed,
        verbosity=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    model_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    model_pred = model.predict(test[FEATURE_COLUMNS])
    model_metrics = classification_metrics(test["label"].to_numpy(), model_pred)
    pr_auc = average_precision_score(test["label"].to_numpy(), model_proba)

    print(f"\nnaive baseline: flag if cancel_rate > {threshold:.4f} (99th pct of train)")
    _print_metrics("naive cancel-rate baseline", baseline_metrics)
    _print_metrics("LightGBM", model_metrics)
    base_rate = float(test["label"].mean())
    print(
        f"{'LightGBM PR-AUC':>28s}  {pr_auc:.4f}  (vs. {base_rate:.5f} for a random ranker "
        f"— base positive rate)"
    )
    print(
        "Note: precision/recall/F1 above are at LightGBM's default 0.5 "
        "probability threshold, which is a poor operating point under this "
        "much class imbalance. PR-AUC is the more honest single-number "
        "summary of ranking quality; a deployed system would pick its "
        "threshold from the precision/recall tradeoff, not use 0.5 as-is."
    )

    print("\nrecall by difficulty tier:")
    print(recall_by_group(test, model_pred, ground_truth, "difficulty_tier").to_string())
    print("\nrecall by pattern type:")
    print(recall_by_group(test, model_pred, ground_truth, "pattern_type").to_string())

    beats_baseline = model_metrics["f1"] > baseline_metrics["f1"]
    print(
        f"\nExit criterion (model beats naive baseline on held-out data): "
        f"{'PASS' if beats_baseline else 'FAIL'} "
        f"(F1 {model_metrics['f1']:.3f} vs {baseline_metrics['f1']:.3f})"
    )

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nfeature importances:")
    for name, imp in importances:
        print(f"  {name:>24s}  {imp}")

    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        model_bin = args.export_dir / "model.bin"
        depth = export_lightgbm_model(model.booster_, FEATURE_COLUMNS, model_bin)
        reference = test[["order_id"] + FEATURE_COLUMNS].copy()
        reference["proba"] = model_proba
        reference["label"] = test["label"].to_numpy()
        reference_path = args.export_dir / "reference_predictions.csv"
        reference.to_csv(reference_path, index=False)
        print(
            f"\nExported {model_bin} (max tree depth {depth}) and "
            f"{reference_path} ({len(reference)} rows) for C++ cross-validation."
        )


if __name__ == "__main__":
    main()
