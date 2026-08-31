import numpy as np
import pandas as pd
import pytest

from python.injection import injector, participants, patterns


def test_background_participants_cover_all_orders_deterministically():
    order_ids = np.array([1, 2, 3, 2, 1, 4])
    a = participants.assign_background_participants(order_ids, num_participants=3, seed=1)
    b = participants.assign_background_participants(order_ids, num_participants=3, seed=1)
    assert a == b
    assert set(a.keys()) == {1, 2, 3, 4}
    for pid in a.values():
        lo, hi = participants.BACKGROUND_PARTICIPANT_RANGE
        assert lo <= pid < hi


def test_manipulator_ids_are_disjoint_from_background():
    lo, hi = participants.MANIPULATOR_PARTICIPANT_RANGE
    assert participants.next_manipulator_id(0) == lo
    with pytest.raises(ValueError):
        participants.next_manipulator_id(hi - lo)


def test_spoofing_pattern_never_executes_and_stays_off_touch():
    rng = np.random.default_rng(0)
    best_bid, best_ask, tick = 100_000, 100_100, 100
    pattern = patterns.generate_spoofing_pattern(
        base_time=10.0, best_bid=best_bid, best_ask=best_ask, tick_size=tick,
        baseline_size=20.0, participant_id=1_000_000, order_id=999, tier="medium", rng=rng,
    )
    assert len(pattern.rows) == 2
    new_row, cancel_row = pattern.rows
    assert new_row["type"] == patterns.NEW_LIMIT_ORDER
    assert cancel_row["type"] == patterns.DELETION
    assert new_row["order_id"] == cancel_row["order_id"] == 999
    assert cancel_row["time"] > new_row["time"]
    # Never crosses the spread, so it can never fill passively.
    if new_row["direction"] == 1:
        assert new_row["price"] < best_bid
    else:
        assert new_row["price"] > best_ask
    # No execution rows anywhere in a spoofing pattern.
    assert all(r["type"] in (patterns.NEW_LIMIT_ORDER, patterns.DELETION) for r in pattern.rows)


def test_layering_pattern_has_non_decreasing_distance_from_touch():
    rng = np.random.default_rng(0)
    best_bid, best_ask, tick = 100_000, 100_100, 100
    order_ids = iter(range(1, 100))
    pattern = patterns.generate_layering_pattern(
        base_time=10.0, best_bid=best_bid, best_ask=best_ask, tick_size=tick,
        baseline_size=20.0, participant_id=1_000_000, order_id_iter=order_ids,
        tier="easy", rng=rng,
    )
    new_rows = [r for r in pattern.rows if r["type"] == patterns.NEW_LIMIT_ORDER]
    cancel_rows = [r for r in pattern.rows if r["type"] == patterns.DELETION]
    assert len(new_rows) >= 3
    assert len(new_rows) == len(cancel_rows)
    assert len({r["order_id"] for r in new_rows}) == len(new_rows)

    side = new_rows[0]["direction"]
    distances = [
        (best_bid - r["price"]) if side == 1 else (r["price"] - best_ask)
        for r in new_rows
    ]
    assert all(later >= earlier for earlier, later in zip(distances, distances[1:]))
    # every submitted order is later cancelled, none executed
    assert {r["order_id"] for r in new_rows} == {r["order_id"] for r in cancel_rows}
    assert pattern.end_time > pattern.start_time


def _toy_message_and_orderbook(n=50):
    rng = np.random.default_rng(42)
    times = np.sort(rng.uniform(34200, 34300, size=n))
    order_ids = np.arange(1, n + 1)
    sizes = rng.integers(1, 50, size=n)
    prices = rng.integers(99000, 101000, size=n)
    directions = rng.choice([1, -1], size=n)
    messages = pd.DataFrame({
        "time": times, "type": 1, "order_id": order_ids,
        "size": sizes, "price": prices, "direction": directions,
    })
    # Minimal 1-level orderbook, repeated to 10-level column count so
    # infer_tick_size's fixed column strides still work.
    row = [100100, 10, 100000, 10] * 10
    orderbook = pd.DataFrame([row] * n)
    return messages, orderbook


def test_inject_end_to_end(tmp_path):
    messages, orderbook = _toy_message_and_orderbook()
    msg_path = tmp_path / "message_10.csv"
    book_path = tmp_path / "orderbook_10.csv"
    messages.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(book_path, header=False, index=False)

    out_dir = tmp_path / "out"
    injector.inject(
        message_csv=msg_path, orderbook_csv=book_path, output_dir=out_dir,
        patterns_per_tier=2, num_background_participants=5, seed=7,
    )

    augmented = pd.read_csv(out_dir / "message_augmented.csv")
    ground_truth = pd.read_csv(out_dir / "ground_truth.csv")

    assert len(ground_truth) == 2 * len(injector.TIERS)
    assert set(ground_truth["difficulty_tier"]) == set(injector.TIERS)
    assert (augmented["time"].diff().dropna() >= 0).all()
    assert augmented["order_id"].is_unique is False  # NEW + DELETE share order_id
    assert augmented["participant_id"].notna().all()

    all_injected_ids = {
        int(oid) for ids in ground_truth["order_ids"] for oid in ids.split(";")
    }
    assert all_injected_ids.issubset(set(augmented["order_id"]))
    # injected order_ids never collide with background order_ids
    assert all_injected_ids.isdisjoint(set(messages["order_id"]))


def test_repeat_manipulator_prob_zero_never_reuses_an_id(tmp_path):
    messages, orderbook = _toy_message_and_orderbook()
    msg_path = tmp_path / "message_10.csv"
    book_path = tmp_path / "orderbook_10.csv"
    messages.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(book_path, header=False, index=False)

    out_dir = tmp_path / "out"
    injector.inject(
        message_csv=msg_path, orderbook_csv=book_path, output_dir=out_dir,
        patterns_per_tier=3, num_background_participants=5,
        repeat_manipulator_prob=0.0, seed=7,
    )
    ground_truth = injector.load_ground_truth(out_dir / "ground_truth.csv")
    assert ground_truth["participant_id"].nunique() == len(ground_truth)


def test_repeat_manipulator_prob_one_reuses_after_the_first_pattern(tmp_path):
    messages, orderbook = _toy_message_and_orderbook()
    msg_path = tmp_path / "message_10.csv"
    book_path = tmp_path / "orderbook_10.csv"
    messages.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(book_path, header=False, index=False)

    out_dir = tmp_path / "out"
    injector.inject(
        message_csv=msg_path, orderbook_csv=book_path, output_dir=out_dir,
        patterns_per_tier=3, num_background_participants=5,
        repeat_manipulator_prob=1.0, seed=7,
    )
    ground_truth = injector.load_ground_truth(out_dir / "ground_truth.csv")
    # every pattern after the first reuses the same one manipulator_id
    assert ground_truth["participant_id"].nunique() == 1


@pytest.mark.timeout(10)
def test_inject_with_single_pattern_per_tier_terminates(tmp_path):
    # Regression: patterns_per_tier=1 means only 3 total patterns, fewer
    # than the 6 (tier, pattern_type) combos — a prior tier-assignment
    # implementation could infinite-loop in this case.
    messages, orderbook = _toy_message_and_orderbook()
    msg_path = tmp_path / "message_10.csv"
    book_path = tmp_path / "orderbook_10.csv"
    messages.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(book_path, header=False, index=False)

    out_dir = tmp_path / "out"
    injector.inject(
        message_csv=msg_path, orderbook_csv=book_path, output_dir=out_dir,
        patterns_per_tier=1, num_background_participants=5, seed=1,
    )
    ground_truth = injector.load_ground_truth(out_dir / "ground_truth.csv")
    assert len(ground_truth) == len(injector.TIERS)
    assert set(ground_truth["difficulty_tier"]) == set(injector.TIERS)


def test_load_ground_truth_keeps_order_ids_as_strings_even_when_all_single_order(tmp_path):
    # Regression: when every pattern is single-order (all cells look like
    # bare integers, no ";"), plain pd.read_csv silently infers the
    # order_ids column as int64 instead of text.
    messages, orderbook = _toy_message_and_orderbook()
    msg_path = tmp_path / "message_10.csv"
    book_path = tmp_path / "orderbook_10.csv"
    messages.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(book_path, header=False, index=False)

    out_dir = tmp_path / "out"
    injector.inject(
        message_csv=msg_path, orderbook_csv=book_path, output_dir=out_dir,
        patterns_per_tier=1, num_background_participants=5, seed=1,
    )
    ground_truth = injector.load_ground_truth(out_dir / "ground_truth.csv")
    assert ground_truth["order_ids"].dtype == object
    for ids in ground_truth["order_ids"]:
        assert isinstance(ids, str)
