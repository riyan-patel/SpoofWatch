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

from sklearn.metrics import average_precision_score

from python.eval.metrics import (
    classification_metrics,
    naive_cancel_rate_baseline_predictions,
    recall_by_group,
    select_threshold_by_macro_f1,
)
from python.injection.injector import load_ground_truth
from python.training.dataset import (
    FEATURE_COLUMNS,
    attach_group_columns,
    build_training_table,
    time_based_split_train_val_test,
)
from python.training.export_model import export_lightgbm_model
from python.training.model import fit_model, scale_pos_weight_for


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
    parser.add_argument(
        "--val-frac", type=float, default=0.2,
        help="chronological slice between train and test used to pick the "
             "deployment threshold, so it's never chosen on the same data "
             "the reported test metrics come from",
    )
    parser.add_argument("--baseline-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="if set, writes model.bin (for the C++ branchless evaluator) and "
             "reference_predictions.csv (test-set features + Python-computed "
             "probability, for cross-validating the C++ side) into this directory",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth_csv)
    table = build_training_table(args.features_csv, args.ground_truth_csv)
    table = attach_group_columns(table, ground_truth)
    train, val, test = time_based_split_train_val_test(
        table, val_frac=args.val_frac, test_frac=args.test_frac
    )

    print(
        f"{len(table)} orders total ({table['label'].sum()} manipulative) — "
        f"{len(train)} train / {len(val)} val / {len(test)} test (time-based split)"
    )
    print(
        f"train positives={train['label'].sum()}  val positives={val['label'].sum()}  "
        f"test positives={test['label'].sum()}"
    )

    baseline_pred, threshold = naive_cancel_rate_baseline_predictions(
        train, test, quantile=args.baseline_quantile
    )
    baseline_metrics = classification_metrics(test["label"].to_numpy(), baseline_pred)

    # `is_unbalance=True` scales the positive class by the full
    # negative:positive ratio, which is ~500-600:1 here. Combined with
    # unregularized leaf-wise growth, that used to blow up raw scores into
    # the hundreds of thousands on some inputs (sigmoid saturating to
    # exactly 1.0) — a handful of feature vectors are shared between
    # positive and negative labels in this dataset (a brand-new
    # participant's very first order looks identical whether it's
    # ordinary or the first leg of an injected pattern), and the model
    # kept fruitlessly re-splitting around that irreducible overlap every
    # boosting round. A capped scale_pos_weight plus leaf/regularization
    # limits (see python/training/model.py) keep raw scores sane without
    # that blowup.
    #
    # (Early stopping on a held-out validation slice was tried instead of
    # the depth/leaf caps below, but with only ~40 validation positives it
    # locked onto a single spurious split — mean_lifetime_ns == 0, a proxy
    # for "brand-new participant" — after one round, collapsing to a
    # 1-tree model with misleadingly perfect metrics. The fixed-size,
    # regularized ensemble below is more robust and, as a side effect,
    # spreads feature importance across the actual behavioral features
    # instead of that one shortcut.)
    model = fit_model(train, seed=args.seed)
    print(f"scale_pos_weight={scale_pos_weight_for(train):.1f}")

    # Threshold is selected on `val` (never seen during fitting) and only
    # then applied to `test`, so the reported test numbers reflect a
    # threshold chosen the way a deployed system would have to choose one
    # — without access to the data it's about to be graded on. Maximizing
    # macro-averaged F1 across pattern_type (rather than one pooled F1)
    # matters even on a single symbol: spoofing patterns are single-order
    # and layering patterns are multi-order, so layering already outweighs
    # spoofing in raw row count here too — pooled F1 could otherwise trade
    # away the whole spoofing class for a marginal layering-precision gain
    # (see python/eval/multi_symbol.py, where pooling across symbols made
    # this failure mode obvious).
    val_proba = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    operating_threshold, val_metrics_at_threshold = select_threshold_by_macro_f1(
        val["label"].to_numpy(), val_proba, val["pattern_type"]
    )

    model_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    model_pred_default = model_proba >= 0.5
    model_pred_operating = model_proba >= operating_threshold
    model_metrics_default = classification_metrics(test["label"].to_numpy(), model_pred_default)
    model_metrics_operating = classification_metrics(test["label"].to_numpy(), model_pred_operating)
    pr_auc = average_precision_score(test["label"].to_numpy(), model_proba)

    print(f"\nnaive baseline: flag if cancel_rate > {threshold:.4f} (99th pct of train)")
    _print_metrics("naive cancel-rate baseline", baseline_metrics)
    _print_metrics("LightGBM @ 0.5 (default)", model_metrics_default)
    print(
        f"operating threshold {operating_threshold:.4f} chosen on val "
        f"(val F1={val_metrics_at_threshold['f1']:.3f})"
    )
    _print_metrics(f"LightGBM @ {operating_threshold:.3f} (chosen)", model_metrics_operating)
    base_rate = float(test["label"].mean())
    print(
        f"{'LightGBM PR-AUC':>28s}  {pr_auc:.4f}  (vs. {base_rate:.5f} for a random ranker "
        f"— base positive rate)"
    )
    print(
        "Note: 0.5 is a poor operating point under this much class "
        "imbalance — it's reported only as a reference. The operating "
        "threshold above is chosen to maximize F1 on a held-out validation "
        "slice (never on test), which is the honest way to pick a real "
        "deployment cutoff; PR-AUC remains the threshold-free ranking-"
        "quality summary."
    )

    print("\nrecall by difficulty tier (at chosen operating threshold):")
    print(recall_by_group(test, model_pred_operating, ground_truth, "difficulty_tier").to_string())
    print("\nrecall by pattern type (at chosen operating threshold):")
    print(recall_by_group(test, model_pred_operating, ground_truth, "pattern_type").to_string())

    beats_baseline = model_metrics_operating["f1"] > baseline_metrics["f1"]
    print(
        f"\nExit criterion (model beats naive baseline on held-out data): "
        f"{'PASS' if beats_baseline else 'FAIL'} "
        f"(F1 {model_metrics_operating['f1']:.3f} vs {baseline_metrics['f1']:.3f})"
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
        threshold_path = args.export_dir / "operating_threshold.txt"
        threshold_path.write_text(f"{operating_threshold:.6f}\n")
        print(
            f"\nExported {model_bin} (max tree depth {depth}), "
            f"{reference_path} ({len(reference)} rows) for C++ cross-validation, "
            f"and {threshold_path} (val-selected deployment threshold)."
        )


if __name__ == "__main__":
    main()
