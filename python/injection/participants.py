"""Synthetic participant-ID assignment for anonymized LOBSTER order flow.

LOBSTER's order_id is just a sequence number with no participant semantics.
To make FeatureEngine's per-participant tracking (order-to-trade ratio,
cancel bursts, layering score, ...) meaningful on real data, every
background order needs a plausible participant_id before we can layer
injected manipulation on top of it.

Real venues have a heavy-tailed population of participants: a handful of
market makers/HFTs generate most of the flow, and a long tail of smaller
participants generate the rest. We approximate that with a Zipf-distributed
pool, assigning participant_id deterministically per order_id (given a
fixed seed) so re-running injection on the same file is reproducible.
"""

from __future__ import annotations

import numpy as np

# Reserve a disjoint participant_id range for synthetic manipulators so
# ground truth can identify them by ID range alone, without cross-referencing
# the ground-truth table.
BACKGROUND_PARTICIPANT_RANGE = (0, 1_000_000)
MANIPULATOR_PARTICIPANT_RANGE = (1_000_000, 2_000_000)


def assign_background_participants(
    order_ids: np.ndarray,
    num_participants: int = 200,
    zipf_exponent: float = 1.3,
    seed: int = 0,
) -> dict[int, int]:
    """Assigns each unique background order_id to a synthetic participant.

    Uses a Zipf-weighted random assignment: participant 0 gets the most
    flow, participant num_participants-1 the least, matching the
    heavy-tailed volume distribution seen in real order flow.
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(order_ids)

    ranks = np.arange(1, num_participants + 1)
    weights = ranks.astype(np.float64) ** (-zipf_exponent)
    weights /= weights.sum()

    chosen = rng.choice(num_participants, size=len(unique_ids), p=weights)
    offset = BACKGROUND_PARTICIPANT_RANGE[0]
    return {int(oid): int(offset + p) for oid, p in zip(unique_ids, chosen)}


def next_manipulator_id(index: int) -> int:
    """Deterministic participant_id for the index-th synthetic manipulator."""
    pid = MANIPULATOR_PARTICIPANT_RANGE[0] + index
    if pid >= MANIPULATOR_PARTICIPANT_RANGE[1]:
        raise ValueError("exceeded manipulator participant_id range")
    return pid
