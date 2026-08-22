"""Synthetic manipulation-pattern generators: spoofing and layering.

Each generator emits a self-contained `Pattern`: a short sequence of
LOBSTER-schema rows (NEW, then DELETE — never an execution, by
construction, since a filled spoof order isn't spoofing) attributed to one
synthetic participant, plus the ground-truth metadata needed to evaluate a
detector against it later.

Difficulty tiers control how far the injected behavior sits from a
plausible legitimate order: "easy" patterns are large, fast, and extreme
relative to background flow; "hard" patterns are sized and timed close
enough to normal activity that a naive threshold rule should miss them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

# Matches spoofwatch::LobsterEventType (cpp/include/spoofwatch/lobster_message.hpp).
NEW_LIMIT_ORDER = 1
DELETION = 3

Tier = Literal["easy", "medium", "hard"]
PatternType = Literal["spoofing", "layering"]

# size_multiplier: range applied to the trailing baseline order size.
# dwell_sec: how long the spoofed/layered orders rest before being pulled.
# num_orders (layering only): how many orders form the stack.
_TIER_PARAMS: dict[Tier, dict] = {
    "easy": {
        "size_multiplier": (8.0, 20.0),
        "dwell_sec": (0.02, 0.15),
        "submit_gap_sec": (0.005, 0.02),
        "num_orders": (5, 7),
    },
    "medium": {
        "size_multiplier": (3.0, 6.0),
        "dwell_sec": (0.2, 0.8),
        "submit_gap_sec": (0.05, 0.2),
        "num_orders": (4, 5),
    },
    "hard": {
        "size_multiplier": (1.2, 2.2),
        "dwell_sec": (0.8, 3.0),
        "submit_gap_sec": (0.2, 0.6),
        "num_orders": (3, 4),
    },
}


@dataclass
class Pattern:
    pattern_type: PatternType
    difficulty_tier: Tier
    participant_id: int
    side: int  # +1 buy-side manipulation, -1 sell-side
    order_ids: list[int]
    start_time: float
    end_time: float
    rows: list[dict] = field(default_factory=list)


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def generate_spoofing_pattern(
    base_time: float,
    best_bid: int,
    best_ask: int,
    tick_size: int,
    baseline_size: float,
    participant_id: int,
    order_id: int,
    tier: Tier,
    rng: np.random.Generator,
) -> Pattern:
    """One large resting order placed just behind the touch, then pulled
    before it can trade. Classic single-order spoofing.
    """
    params = _TIER_PARAMS[tier]
    side = int(rng.choice([1, -1]))
    size = max(1, int(round(baseline_size * _uniform(rng, *params["size_multiplier"]))))
    dwell = _uniform(rng, *params["dwell_sec"])

    # One tick behind the current touch on the manipulated side: visible,
    # priority-adjacent, but not crossing (so it never fills passively).
    price = int(best_bid - tick_size) if side == 1 else int(best_ask + tick_size)

    new_row = {
        "time": base_time,
        "type": NEW_LIMIT_ORDER,
        "order_id": order_id,
        "size": size,
        "price": price,
        "direction": side,
        "participant_id": participant_id,
    }
    cancel_row = {
        "time": base_time + dwell,
        "type": DELETION,
        "order_id": order_id,
        "size": size,
        "price": price,
        "direction": side,
        "participant_id": participant_id,
    }
    return Pattern(
        pattern_type="spoofing",
        difficulty_tier=tier,
        participant_id=participant_id,
        side=side,
        order_ids=[order_id],
        start_time=base_time,
        end_time=cancel_row["time"],
        rows=[new_row, cancel_row],
    )


def generate_layering_pattern(
    base_time: float,
    best_bid: int,
    best_ask: int,
    tick_size: int,
    baseline_size: float,
    participant_id: int,
    order_id_iter: Iterator[int],
    tier: Tier,
    rng: np.random.Generator,
) -> Pattern:
    """A stack of same-side orders at non-decreasing distance from the
    touch, submitted in quick succession and pulled together shortly after
    — matches FeatureEngine's layering_score definition exactly, since
    that's the behavior it's designed to catch.
    """
    params = _TIER_PARAMS[tier]
    side = int(rng.choice([1, -1]))
    num_orders = int(rng.integers(params["num_orders"][0], params["num_orders"][1] + 1))

    order_ids: list[int] = []
    new_rows: list[dict] = []
    t = base_time
    distance = tick_size
    for i in range(num_orders):
        oid = next(order_id_iter)
        order_ids.append(oid)
        size = max(1, int(round(baseline_size * _uniform(rng, *params["size_multiplier"]))))
        price = int(best_bid - distance) if side == 1 else int(best_ask + distance)
        new_rows.append({
            "time": t,
            "type": NEW_LIMIT_ORDER,
            "order_id": oid,
            "size": size,
            "price": price,
            "direction": side,
            "participant_id": participant_id,
        })
        # Non-decreasing distance from touch, strictly increasing here to
        # keep every consecutive pair inside the layering window.
        distance += tick_size * int(rng.integers(1, 3))
        t += _uniform(rng, *params["submit_gap_sec"])

    dwell = _uniform(rng, *params["dwell_sec"])
    cancel_start = new_rows[-1]["time"] + dwell
    cancel_rows: list[dict] = []
    ct = cancel_start
    # Cancel burst: pull the whole stack in reverse order, tightly spaced.
    for row in reversed(new_rows):
        cancel_rows.append({
            "time": ct,
            "type": DELETION,
            "order_id": row["order_id"],
            "size": row["size"],
            "price": row["price"],
            "direction": row["direction"],
            "participant_id": participant_id,
        })
        ct += _uniform(rng, *params["submit_gap_sec"]) * 0.5

    rows = new_rows + cancel_rows
    return Pattern(
        pattern_type="layering",
        difficulty_tier=tier,
        participant_id=participant_id,
        side=side,
        order_ids=order_ids,
        start_time=base_time,
        end_time=cancel_rows[-1]["time"],
        rows=rows,
    )
