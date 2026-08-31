"""Phase 3 pipeline: inject synthetic spoofing/layering patterns into a
clean LOBSTER message file, producing a participant-labeled event stream
plus a separate ground-truth table.

Clean LOBSTER data is treated as non-manipulative background flow. Every
background order gets a synthetic participant_id (participants.py) so
per-participant features are meaningful; then, at a set of injection
points spread through the day, one of a handful of synthetic manipulators
executes a spoofing or layering pattern (patterns.py) using the real
top-of-book prices at that moment (read from LOBSTER's own orderbook
reference file, so injected orders sit at realistic, non-crossing prices).

Ground truth (order_id -> pattern_type/tier/participant) is tracked
separately from the augmented message stream, matching the "ground truth
tracked separately" exit criterion in docs/PHASES.md.

Usage:
    python -m python.injection.injector \\
        data/lobster_samples/AAPL_2012-06-21_10/message_10.csv \\
        data/lobster_samples/AAPL_2012-06-21_10/orderbook_10.csv \\
        data/synthetic/AAPL_2012-06-21 \\
        --patterns-per-tier 15 --seed 0
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from python.injection import participants as participants_mod
from python.injection import patterns as patterns_mod

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]
AUGMENTED_COLUMNS = MESSAGE_COLUMNS + ["participant_id"]
GROUND_TRUTH_COLUMNS = [
    "pattern_id", "pattern_type", "difficulty_tier", "participant_id",
    "side", "order_ids", "start_time", "end_time",
]

TIERS: list[patterns_mod.Tier] = ["easy", "medium", "hard"]
PATTERN_TYPES: list[patterns_mod.PatternType] = ["spoofing", "layering"]


def load_messages(message_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(message_csv, header=None, names=MESSAGE_COLUMNS)
    df["price"] = df["price"].astype(np.int64)
    df["order_id"] = df["order_id"].astype(np.int64)
    return df


def load_ground_truth(ground_truth_csv: Path) -> pd.DataFrame:
    """Reads back a ground_truth.csv written by `inject`.

    order_ids must be read as a string column: it's a ";"-joined list of
    IDs, but when every pattern in a batch happens to be single-order
    (e.g. all spoofing, no layering), every cell is a single bare integer
    with no ";", and pandas' automatic dtype inference silently coerces
    the whole column to int64 instead of leaving it as text. Always go
    through this helper rather than calling pd.read_csv directly.
    """
    return pd.read_csv(ground_truth_csv, dtype={"order_ids": str})


def load_orderbook(orderbook_csv: Path) -> pd.DataFrame:
    return pd.read_csv(orderbook_csv, header=None)


def infer_tick_size(book_row: pd.Series, default: int = 100) -> int:
    ask_prices = book_row.iloc[0::4].to_numpy(dtype=np.int64)
    bid_prices = book_row.iloc[2::4].to_numpy(dtype=np.int64)
    diffs = np.concatenate([np.diff(np.sort(np.unique(ask_prices))),
                             np.diff(np.sort(np.unique(bid_prices)))])
    diffs = diffs[diffs > 0]
    return int(diffs.min()) if len(diffs) else default


def _choose_injection_points(
    n_messages: int,
    total_patterns: int,
    min_gap: int,
    margin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evenly-spaced-ish injection indices with jitter, far enough from the
    file's edges and from each other that patterns don't overlap.
    """
    usable_lo, usable_hi = margin, n_messages - margin
    span = usable_hi - usable_lo
    if total_patterns * min_gap >= span:
        raise ValueError("not enough room in the message file for the requested pattern count")

    slot = span / total_patterns
    points = []
    for i in range(total_patterns):
        lo = usable_lo + i * slot
        hi = usable_lo + (i + 1) * slot
        points.append(int(rng.uniform(lo, hi)))
    return np.array(points, dtype=np.int64)


def inject(
    message_csv: Path,
    orderbook_csv: Path,
    output_dir: Path,
    patterns_per_tier: int = 15,
    num_background_participants: int = 200,
    repeat_manipulator_prob: float = 0.0,
    seed: int = 0,
) -> None:
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
    next_synthetic_order_id = int(messages["order_id"].max()) + 1
    order_id_iter = itertools.count(next_synthetic_order_id)

    total_patterns = patterns_per_tier * len(TIERS)
    margin = min(100, max(1, len(messages) // 10))
    injection_points = _choose_injection_points(
        n_messages=len(messages),
        total_patterns=total_patterns,
        min_gap=1,
        margin=margin,
        rng=rng,
    )
    injection_points.sort()

    # Exactly patterns_per_tier entries per tier, pattern_type alternating
    # within each tier, then shuffled together so tiers/types are
    # interleaved through the day rather than blocked together.
    final_assignment = [
        (tier, PATTERN_TYPES[i % len(PATTERN_TYPES)])
        for tier in TIERS
        for i in range(patterns_per_tier)
    ]
    rng.shuffle(final_assignment)

    # Every pattern gets a brand-new manipulator_id by default, so the
    # model never sees a *second* manipulative act from an identity with
    # real rolling history — Phase 7's Coscia case study found this makes
    # the model partly learn "this participant's history is empty" as a
    # manipulation proxy instead of the cancel-heavy behavior itself. With
    # repeat_manipulator_prob > 0, some patterns reuse an already-used
    # manipulator_id instead of minting a new one, so training data
    # includes repeat offenders whose own history is no longer empty by
    # the time they manipulate again.
    used_manipulator_pids: list[int] = []
    next_new_manipulator_index = 0

    patterns: list[patterns_mod.Pattern] = []
    for idx, (tier, ptype) in zip(injection_points, final_assignment):
        row = messages.iloc[idx]
        book_row = orderbook.iloc[idx]
        best_ask, best_bid = int(book_row.iloc[0]), int(book_row.iloc[2])
        tick_size = infer_tick_size(book_row)
        base_time = float(row["time"]) + 1e-6

        if used_manipulator_pids and rng.random() < repeat_manipulator_prob:
            manipulator_pid = int(rng.choice(used_manipulator_pids))
        else:
            manipulator_pid = participants_mod.next_manipulator_id(next_new_manipulator_index)
            next_new_manipulator_index += 1
            used_manipulator_pids.append(manipulator_pid)

        if ptype == "spoofing":
            pattern = patterns_mod.generate_spoofing_pattern(
                base_time=base_time,
                best_bid=best_bid,
                best_ask=best_ask,
                tick_size=tick_size,
                baseline_size=baseline_size,
                participant_id=manipulator_pid,
                order_id=next(order_id_iter),
                tier=tier,
                rng=rng,
            )
        else:
            pattern = patterns_mod.generate_layering_pattern(
                base_time=base_time,
                best_bid=best_bid,
                best_ask=best_ask,
                tick_size=tick_size,
                baseline_size=baseline_size,
                participant_id=manipulator_pid,
                order_id_iter=order_id_iter,
                tier=tier,
                rng=rng,
            )
        patterns.append(pattern)

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
        }
        for i, p in enumerate(patterns)
    ], columns=GROUND_TRUTH_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_dir / "message_augmented.csv", index=False)
    ground_truth.to_csv(output_dir / "ground_truth.csv", index=False)
    print(
        f"Injected {len(patterns)} patterns "
        f"({patterns_per_tier} per tier x {len(TIERS)} tiers) into "
        f"{len(messages)} background events -> {len(augmented)} total rows.\n"
        f"Wrote {output_dir / 'message_augmented.csv'} and "
        f"{output_dir / 'ground_truth.csv'}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message_csv", type=Path)
    parser.add_argument("orderbook_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--patterns-per-tier", type=int, default=15)
    parser.add_argument("--num-background-participants", type=int, default=200)
    parser.add_argument(
        "--repeat-manipulator-prob", type=float, default=0.0,
        help="probability a pattern reuses an already-used manipulator_id "
             "instead of minting a new one, so training data includes "
             "repeat offenders (see python/eval/case_study.py)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inject(
        message_csv=args.message_csv,
        orderbook_csv=args.orderbook_csv,
        output_dir=args.output_dir,
        patterns_per_tier=args.patterns_per_tier,
        num_background_participants=args.num_background_participants,
        repeat_manipulator_prob=args.repeat_manipulator_prob,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
