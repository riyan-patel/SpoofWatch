import numpy as np
import pandas as pd

from python.eval.case_study import inject_coscia_style_case_study
from python.injection import participants


def _toy_message_and_orderbook(n=200):
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
    row = [100100, 10, 100000, 10] * 10
    orderbook = pd.DataFrame([row] * n)
    return messages, orderbook


def test_coscia_case_study_injects_one_manipulator_repeated_cycles(tmp_path):
    messages, orderbook = _toy_message_and_orderbook()
    msg_path = tmp_path / "message_10.csv"
    book_path = tmp_path / "orderbook_10.csv"
    messages.to_csv(msg_path, header=False, index=False)
    orderbook.to_csv(book_path, header=False, index=False)

    out_dir = tmp_path / "out"
    num_cycles = inject_coscia_style_case_study(
        msg_path, book_path, out_dir, num_cycles=6, num_background_participants=5, seed=3,
    )
    assert num_cycles == 6

    augmented = pd.read_csv(out_dir / "message_augmented.csv")
    ground_truth = pd.read_csv(out_dir / "ground_truth.csv", dtype={"order_ids": str})

    assert len(ground_truth) == 6
    assert (ground_truth["pattern_type"] == "spoofing").all()
    # every cycle comes from the same synthetic manipulator, unlike the
    # regular injector which mints a fresh participant_id per pattern.
    assert ground_truth["participant_id"].nunique() == 1
    assert ground_truth["participant_id"].iloc[0] == participants.next_manipulator_id(0)
    assert list(ground_truth["cycle_index"]) == list(range(6))

    # cycles fire back-to-back, not spread across the whole file
    span = ground_truth["end_time"].max() - ground_truth["start_time"].min()
    assert span < (messages["time"].max() - messages["time"].min())

    injected_ids = {int(x) for cell in ground_truth["order_ids"] for x in cell.split(";")}
    assert injected_ids.issubset(set(augmented["order_id"]))
    assert injected_ids.isdisjoint(set(messages["order_id"]))
    assert (augmented["time"].diff().dropna() >= 0).all()
