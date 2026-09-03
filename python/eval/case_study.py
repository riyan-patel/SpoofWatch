"""Phase 7 qualitative sanity check: does the detector flag behavior that
looks like a publicly documented real spoofing case, not just the
synthetic patterns it was trained to catch?

The reference case is Michael Coscia / Panther Energy Trading (CFTC
enforcement action 2013, first criminal spoofing conviction under
Dodd-Frank 2015): an algorithm that repeatedly placed a large order on one
side of the book — visible, priority-adjacent, never intended to fill —
then cancelled it within a fraction of a second, thousands of times a day,
alternating sides. That is exactly the shape `patterns.generate_
spoofing_pattern`'s "easy" tier already produces (large order one tick
behind the touch, pulled after a short dwell); what's novel here is firing
it in a tight, repeated burst from one participant rather than a single
isolated injection, since Coscia's documented behavior was its
repetitiveness, not any one order in isolation.

This is a sanity check, not a benchmark: one real trading day, one
synthetic participant, no claim of statistical significance. It answers a
narrower question — does the trained model keep flagging this participant
consistently across a repeated burst, or does it only catch the first
cycle and then miss the rest (e.g. because per-participant rolling
features saturate or reset in a way that hides sustained manipulation)?

Usage:
    python -m python.eval.case_study \\
        data/lobster_samples/AAPL_2012-06-21_10/message_10.csv \\
        data/lobster_samples/AAPL_2012-06-21_10/orderbook_10.csv \\
        data/synthetic/AAPL_2012-06-21/features.csv \\
        data/synthetic/AAPL_2012-06-21/ground_truth.csv \\
        data/synthetic/coscia_case_study \\
        --num-cycles 15 --seed 1
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from python.eval.metrics import select_threshold_by_macro_f1
from python.injection import participants as participants_mod
from python.injection import patterns as patterns_mod
from python.injection.injector import (
    AUGMENTED_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    infer_tick_size,
    load_ground_truth,
    load_messages,
    load_orderbook,
)
from python.training.dataset import (
    FEATURE_COLUMNS,
    attach_group_columns,
    build_training_table,
    time_based_split_train_val_test,
)
from python.training.model import fit_model


def inject_coscia_style_case_study(
    message_csv: Path,
    orderbook_csv: Path,
    output_dir: Path,
    num_cycles: int = 15,
    num_background_participants: int = 200,
    seed: int = 0,
) -> int:
    """Injects `num_cycles` repeated spoofing bursts from a single
    synthetic manipulator at one anchor point in the file, alternating
    sides — the repeated-and-rapid cadence that distinguishes the Coscia
    case from an isolated one-off spoof.
    """
    rng = np.random.default_rng(seed)

    messages = load_messages(message_csv)
    orderbook = load_orderbook(orderbook_csv)
    if len(messages) != len(orderbook):
        raise ValueError(
            f"message file ({len(messages)} rows) and orderbook file "
            f"({len(orderbook)} rows) are not aligned"
        )

    background_map = participants_mod.assign_background_participants(
        messages["order_id"].to_numpy(),
        num_participants=num_background_participants,
        seed=seed,
    )
    messages["participant_id"] = messages["order_id"].map(background_map)

    baseline_size = float(messages["size"].median())
    order_id_iter = itertools.count(int(messages["order_id"].max()) + 1)
    manipulator_pid = participants_mod.next_manipulator_id(0)

    # One anchor point (mid-file, well clear of the open/close) whose
    # real top-of-book prices seed every cycle in the burst — the whole
    # burst plays out over a few seconds, short enough that reusing one
    # book snapshot is the same approximation the regular injector makes
    # per-pattern.
    anchor_idx = len(messages) // 2
    anchor_row = messages.iloc[anchor_idx]
    book_row = orderbook.iloc[anchor_idx]
    best_ask, best_bid = int(book_row.iloc[0]), int(book_row.iloc[2])
    tick_size = infer_tick_size(book_row)

    patterns: list[patterns_mod.Pattern] = []
    t = float(anchor_row["time"]) + 1e-6
    for _ in range(num_cycles):
        pattern = patterns_mod.generate_spoofing_pattern(
            base_time=t,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_size=tick_size,
            baseline_size=baseline_size,
            participant_id=manipulator_pid,
            order_id=next(order_id_iter),
            tier="easy",
            rng=rng,
        )
        patterns.append(pattern)
        # Next cycle fires shortly after the last one's cancel — Coscia's
        # documented cadence was thousands of these a day, i.e. back to
        # back within a small multiple of a second, not spread evenly
        # across the whole trading day like the standard injector's
        # patterns.
        t = pattern.end_time + float(rng.uniform(0.05, 0.5))

    injected_rows = pd.DataFrame(
        [r for p in patterns for r in p.rows], columns=AUGMENTED_COLUMNS
    )
    augmented = pd.concat([messages, injected_rows], ignore_index=True)
    augmented = augmented.sort_values("time", kind="mergesort").reset_index(drop=True)

    ground_truth = pd.DataFrame([
        {
            "pattern_id": i,
            "pattern_type": p.pattern_type,
            "difficulty_tier": p.difficulty_tier,
            "participant_id": p.participant_id,
            "side": p.side,
            "order_ids": ";".join(str(oid) for oid in p.order_ids),
            "start_time": p.start_time,
            "end_time": p.end_time,
            "cycle_index": i,
        }
        for i, p in enumerate(patterns)
    ], columns=GROUND_TRUTH_COLUMNS + ["cycle_index"])

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_dir / "message_augmented.csv", index=False)
    ground_truth.to_csv(output_dir / "ground_truth.csv", index=False)
    return len(patterns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message_csv", type=Path)
    parser.add_argument("orderbook_csv", type=Path)
    parser.add_argument(
        "training_features_csv", type=Path,
        help="features.csv from a regular (non-case-study) injection run, "
             "used to fit the scoring model and pick its operating threshold",
    )
    parser.add_argument(
        "training_ground_truth_csv", type=Path,
        help="ground_truth.csv matching training_features_csv",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-cycles", type=int, default=15)
    parser.add_argument("--num-background-participants", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--spoofwatch-features-bin", type=Path, default=Path("build/spoofwatch_features"),
        help="compiled C++ binary that replays a message file through the "
             "real OrderBook + FeatureEngine (same engine used for training)",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    num_cycles = inject_coscia_style_case_study(
        args.message_csv,
        args.orderbook_csv,
        args.output_dir,
        num_cycles=args.num_cycles,
        num_background_participants=args.num_background_participants,
        seed=args.seed,
    )
    print(f"Injected a {num_cycles}-cycle Coscia-style spoofing burst.")

    case_study_features_csv = args.output_dir / "features.csv"
    subprocess.run(
        [
            str(args.spoofwatch_features_bin),
            str(args.output_dir / "message_augmented.csv"),
            str(case_study_features_csv),
        ],
        check=True,
    )

    # Fit the scoring model exactly the way train.py would, on the
    # regular (unrelated) injection run passed in — this case study asks
    # whether a model trained the normal way generalizes to a differently
    # -shaped real-world pattern, not whether a model can be tuned to
    # this specific burst.
    training_ground_truth = load_ground_truth(args.training_ground_truth_csv)
    training_table = build_training_table(args.training_features_csv, args.training_ground_truth_csv)
    training_table = attach_group_columns(training_table, training_ground_truth)
    train, val, _test = time_based_split_train_val_test(
        training_table, val_frac=args.val_frac, test_frac=args.test_frac
    )
    model = fit_model(train, seed=args.seed)
    val_proba = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    threshold, _ = select_threshold_by_macro_f1(val["label"].to_numpy(), val_proba, val["pattern_type"])
    print(f"Scoring with operating threshold {threshold:.4f} (selected on the regular run's val split)")

    case_study_table = build_training_table(case_study_features_csv, args.output_dir / "ground_truth.csv")
    case_study_table["proba"] = model.predict_proba(case_study_table[FEATURE_COLUMNS])[:, 1]

    cycles = case_study_table[case_study_table["label"] == 1].sort_values("time")
    background = case_study_table[case_study_table["label"] == 0]

    print(f"\n{len(cycles)}/{num_cycles} burst orders scored (one per cycle):")
    flagged = 0
    for i, (_, row) in enumerate(cycles.iterrows()):
        hit = row["proba"] >= threshold
        flagged += int(hit)
        print(f"  cycle {i:>2d}  proba={row['proba']:.4f}  {'FLAGGED' if hit else 'missed'}")

    recall = flagged / len(cycles) if len(cycles) else 0.0
    background_flag_rate = float((background["proba"] >= threshold).mean())
    print(
        f"\nBurst recall at operating threshold: {recall:.3f} ({flagged}/{len(cycles)} cycles)"
    )
    print(
        f"Background false-positive rate on this same file: {background_flag_rate:.5f} "
        f"({int((background['proba'] >= threshold).sum())}/{len(background)} ordinary orders)"
    )

    if len(cycles) > 1:
        first_hit = bool(cycles.iloc[0]["proba"] >= threshold)
        later_recall = float((cycles.iloc[1:]["proba"] >= threshold).mean())
        if first_hit and later_recall < recall:
            print(
                f"\nObservation: recall on cycles after the first ({later_recall:.3f}) is "
                f"lower than overall burst recall ({recall:.3f}) — this manipulator is "
                "caught less reliably once it has some order history of its own. If the "
                "training run's injection used repeat_manipulator_prob=0.0 (the "
                "default), the training data never contains a *second* manipulative act "
                "from the same identity, and the model may have partly learned 'this "
                "participant's history is empty' as a manipulation proxy rather than "
                "the cancel-heavy behavior itself — see docs/PHASES.md Phase 4's "
                "feature-importance caveat. Re-run the training injection with "
                "--repeat-manipulator-prob > 0 (python/injection/injector.py) to give "
                "the model repeat-offender examples to learn from."
            )
    print(
        "\nThis is a qualitative sanity check against one publicly documented "
        "case's known modus operandi, not a quantitative benchmark — see "
        "docs/PHASES.md Phase 7 and README.md's Ground truth section."
    )


if __name__ == "__main__":
    main()
