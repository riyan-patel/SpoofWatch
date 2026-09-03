"""Phase 8: pools synthetic injection + evaluation across several LOBSTER
symbols instead of the single AAPL/2012-06-21 day used everywhere else in
this project.

Every number reported elsewhere (Phase 4's precision/recall, Phase 7's
threshold selection) comes from one stock, one trading day, a few hundred
injected positives — explicitly flagged throughout as noisy and
sample-size-sensitive. The training/injection code itself was already
symbol-agnostic (it just takes file paths), so the only thing missing was
an orchestration layer that runs the same pipeline across multiple
datasets and reports pooled + per-symbol numbers, instead of the same
handful of positives from one file.

Datasets are auto-discovered from `data/lobster_samples/<TICKER>_<DATE>_
<LEVEL>/{message,orderbook}_<LEVEL>.csv`, the same naming convention
`injector.py`'s own usage examples already assume.

Each symbol is injected and split (train/val/test) chronologically on its
own — splitting "by time" across different symbols on the same calendar
day wouldn't mean anything — and then the per-symbol splits are pooled:
one model is fit on the union of all symbols' train slices, the operating
threshold is chosen on the union of val slices, and final metrics are
reported on the union of test slices, both overall and broken out by
symbol.

Usage:
    python -m python.eval.multi_symbol --patterns-per-tier 40 \\
        --repeat-manipulator-prob 0.3 --seed 0
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from python.eval.metrics import classification_metrics, select_threshold_by_macro_f1
from python.injection import injector
from python.training.dataset import FEATURE_COLUMNS, attach_group_columns, build_training_table
from python.training.model import fit_model

_DATASET_DIR_RE = re.compile(r"^(?P<ticker>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<level>\d+)$")


@dataclass
class LobsterDataset:
    ticker: str
    date: str
    level: int
    message_csv: Path
    orderbook_csv: Path


def discover_lobster_samples(root: Path) -> list[LobsterDataset]:
    """Finds every `<TICKER>_<DATE>_<LEVEL>/{message,orderbook}_<LEVEL>.csv`
    pair under `root`, sorted by ticker so runs are reproducible.
    """
    datasets = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _DATASET_DIR_RE.match(entry.name)
        if not match:
            continue
        level = int(match.group("level"))
        message_csv = entry / f"message_{level}.csv"
        orderbook_csv = entry / f"orderbook_{level}.csv"
        if message_csv.exists() and orderbook_csv.exists():
            datasets.append(LobsterDataset(
                ticker=match.group("ticker"), date=match.group("date"), level=level,
                message_csv=message_csv, orderbook_csv=orderbook_csv,
            ))
    return datasets


def build_symbol_table(
    dataset: LobsterDataset,
    output_root: Path,
    patterns_per_tier: int,
    repeat_manipulator_prob: float,
    seed: int,
    spoofwatch_features_bin: Path,
) -> pd.DataFrame:
    output_dir = output_root / f"{dataset.ticker}_{dataset.date}"
    injector.inject(
        message_csv=dataset.message_csv,
        orderbook_csv=dataset.orderbook_csv,
        output_dir=output_dir,
        patterns_per_tier=patterns_per_tier,
        repeat_manipulator_prob=repeat_manipulator_prob,
        seed=seed,
    )

    features_csv = output_dir / "features.csv"
    subprocess.run(
        [str(spoofwatch_features_bin), str(output_dir / "message_augmented.csv"), str(features_csv)],
        check=True,
    )

    ground_truth_csv = output_dir / "ground_truth.csv"
    table = build_training_table(features_csv, ground_truth_csv)
    table = attach_group_columns(table, injector.load_ground_truth(ground_truth_csv))
    table["symbol"] = dataset.ticker
    return table.sort_values("time").reset_index(drop=True)


def _split_three_way(table: pd.DataFrame, val_frac: float, test_frac: float):
    n = len(table)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    return table.iloc[:train_end], table.iloc[train_end:val_end], table.iloc[val_end:]


def _print_metrics(name: str, m: dict) -> None:
    print(
        f"{name:>28s}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
        f"f1={m['f1']:.3f}  fpr={m['fpr']:.4f}  "
        f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lobster-samples-dir", type=Path, default=Path("data/lobster_samples"),
        help="root to auto-discover <TICKER>_<DATE>_<LEVEL> dataset folders under",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="restrict to these tickers (default: every discovered dataset)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/synthetic/multi_symbol"))
    parser.add_argument("--patterns-per-tier", type=int, default=40)
    parser.add_argument("--repeat-manipulator-prob", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--spoofwatch-features-bin", type=Path, default=Path("build/spoofwatch_features"),
    )
    args = parser.parse_args()

    datasets = discover_lobster_samples(args.lobster_samples_dir)
    if args.symbols:
        wanted = set(args.symbols)
        datasets = [d for d in datasets if d.ticker in wanted]
    if not datasets:
        raise SystemExit(f"no LOBSTER datasets found under {args.lobster_samples_dir}")

    print(f"Discovered {len(datasets)} dataset(s): " + ", ".join(f"{d.ticker} ({d.date}, level {d.level})" for d in datasets))

    train_parts, val_parts, test_parts = [], [], []
    for dataset in datasets:
        table = build_symbol_table(
            dataset, args.output_root, args.patterns_per_tier,
            args.repeat_manipulator_prob, args.seed, args.spoofwatch_features_bin,
        )
        train, val, test = _split_three_way(table, args.val_frac, args.test_frac)
        print(
            f"  {dataset.ticker:>6s}: {len(table)} orders ({table['label'].sum()} manipulative) "
            f"-> {len(train)} train / {len(val)} val / {len(test)} test"
        )
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    print(
        f"\nPooled across {len(datasets)} symbols: {len(train) + len(val) + len(test)} orders total "
        f"({train['label'].sum() + val['label'].sum() + test['label'].sum()} manipulative) — "
        f"{len(train)} train / {len(val)} val / {len(test)} test"
    )

    model = fit_model(train, seed=args.seed)
    val_proba = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    # Macro-averaged across pattern_type, not plain pooled F1: pooling
    # multiple symbols means layering (multi-order) positives vastly
    # outnumber spoofing (single-order) ones in raw row count, and a
    # pooled-F1-maximizing threshold can zero out the whole minority
    # pattern type for a marginal gain on the majority one. See
    # select_threshold_by_macro_f1's docstring — this was found, not
    # assumed, on this exact pooled multi-symbol run.
    threshold, _ = select_threshold_by_macro_f1(
        val["label"].to_numpy(), val_proba, val["pattern_type"]
    )

    test_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_pred = test_proba >= threshold
    metrics = classification_metrics(test["label"].to_numpy(), test_pred)
    pr_auc = average_precision_score(test["label"].to_numpy(), test_proba)

    print(f"\noperating threshold {threshold:.4f} (chosen on pooled val)")
    _print_metrics("pooled LightGBM", metrics)
    print(f"{'pooled PR-AUC':>28s}  {pr_auc:.4f}")

    print("\nrecall by symbol (at pooled operating threshold):")
    positives = test[test["label"] == 1].copy()
    positives["_pred"] = test_pred[test["label"].to_numpy() == 1]
    print(positives.groupby("symbol")["_pred"].mean().rename("recall").to_string())

    print("\nrecall by difficulty tier (at pooled operating threshold):")
    print(positives.groupby("difficulty_tier")["_pred"].mean().rename("recall").to_string())

    print("\nrecall by pattern type (at pooled operating threshold):")
    print(positives.groupby("pattern_type")["_pred"].mean().rename("recall").to_string())

    print(
        "\nThis pools evaluation across multiple symbols/files instead of "
        "one trading day's few hundred positives — still a single day per "
        "symbol, not multiple days, so it widens the sample rather than "
        "eliminating the small-sample caveat entirely."
    )


if __name__ == "__main__":
    main()
