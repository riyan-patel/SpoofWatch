# SpoofWatch

Low-latency C++ order-flow surveillance engine with inline ML-based spoofing/layering detection.


## Problem

Exchanges and trading firms need to detect manipulative order-flow patterns —
**spoofing** (large orders placed with no intent to execute) and **layering**
(stacked fake orders faking supply/demand pressure) — as they happen, not in
an overnight batch report. This project builds a system that:

1. Ingests a live (simulated) order-by-order market data feed
2. Reconstructs the limit order book and tracks individual order lifecycles in real time
3. Computes manipulation-indicative features incrementally, with no wasted recomputation
4. Runs ML inference inline to flag suspicious behavior within microseconds
5. Reports precision/recall against a synthetic ground truth, plus a full latency breakdown

The goal isn't "detect spoofing perfectly" — no one can, with public data. It's
demonstrating a real-time, low-latency system that fuses market microstructure
understanding with inline ML inference, with an honest accounting of where the
ground-truth labels actually come from.

## Architecture

```
Raw ITCH-style   →  [1] Feed Handler  →  [2] Order Lifecycle Tracker
message stream          |                         |
(LOBSTER data)           v                         v
                   Order Book State        Per-Participant Rolling
                   (bid/ask ladders)        Feature Windows
                          \                       /
                           v                     v
                          [3] Feature Vector
                                  v
                    [4] ML Inference (branchless GBT evaluator)
                                  v
                    [5] Alert Emitter (flag + evidence log)

     Instrumentation: per-stage latency histograms throughout
```

**Design rule that matters most:** nothing in stages 1–4 allocates on the heap
or blocks. Every data structure is fixed-size, pre-allocated at startup.

## Tech stack

| Layer | Choice |
|---|---|
| Core engine | C++20, CMake, GoogleTest |
| Order tracking | Custom open-addressing hash map, no `std::unordered_map` in hot path |
| Book structure | Fixed-size price-level array / intrusive list, no `std::map` |
| Feature windows | Fixed-capacity ring buffers, incremental updates only |
| Model training | Python 3.11, LightGBM (+ optional PyTorch autoencoder) |
| Hot-path inference | Hand-rolled branchless tree evaluator (`TreeModel`, see below) |
| Data source | [LOBSTER](https://lobsterdata.com) message-level L3 data |
| Benchmarking | `perf`, `clock_gettime(CLOCK_MONOTONIC)`, flame graphs |

## Repo layout

```
cpp/
  include/spoofwatch/   public headers
  src/                  engine implementation
  tests/                GoogleTest suite
python/
  injection/            synthetic manipulation-pattern injection
  training/             LightGBM / model training
  eval/                 precision/recall/FPR evaluation
data/
  lobster_samples/      raw LOBSTER data (gitignored, download locally)
  synthetic/            generated ground-truth labeled data (gitignored)
docs/
  PHASES.md             phase-by-phase build plan and exit criteria
benchmarks/
  results/              latency histograms, flamegraphs (gitignored)
```

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spoofwatch
ctest --test-dir build

# Replay a LOBSTER message file through OrderBook and diff against LOBSTER's
# own reference orderbook output (see "Order book reconstruction" below for
# why this doesn't reach 100%):
./build/spoofwatch_validate_book \
  data/lobster_samples/AAPL_2012-06-21_10/message_10.csv \
  data/lobster_samples/AAPL_2012-06-21_10/orderbook_10.csv \
  10

# Replay an injected message stream through OrderBook + FeatureEngine and
# dump one feature-vector row per order lifecycle event (see "Model
# training" below):
./build/spoofwatch_features \
  data/synthetic/AAPL_2012-06-21/message_augmented.csv \
  data/synthetic/AAPL_2012-06-21/features.csv
```

## Order book reconstruction

`OrderBook` (order pool + per-side price-level books, see `cpp/include/spoofwatch/`)
replays a LOBSTER message stream and maintains live bid/ask depth with zero
heap allocation past construction. `spoofwatch_validate_book` diffs the
result against LOBSTER's own reference `orderbook_*.csv` row by row.

That diff won't reach 100% on a depth-limited file, and that's expected, not
a bug: LOBSTER's message file only records events for orders that affect the
requested top-N price levels *at the time of the event*. An order can start
inside the top 10, get buried deeper as later orders arrive, and then be
cancelled or executed with no message ever appearing in a 10-level file —
by then it's no longer a top-10-affecting event. Separately, the very first
row already reflects the opening auction's effect on the book, but the
auction event itself (`type 6`) is stripped from LOBSTER's public "clean"
samples, so the message file can't explain that starting depth either.
Verified this isn't a mirror/corruption issue: two independent copies of
the AAPL 2012-06-21 sample are byte-identical, and an independent
from-scratch Python replay reproduces the same match statistics as the C++
engine. Core per-event-type logic (new/cancel/delete/execute) is verified
directly with hand-computed toy sequences in `cpp/tests/order_book_test.cpp`.

## Feature engine

`FeatureEngine` (`cpp/include/spoofwatch/feature_engine.hpp`) tracks
per-participant rolling features incrementally — never recomputed from
history — using `RingBuffer` (fixed-capacity circular buffer) and
`IncrementalStats` (Welford's online mean/variance):

- order-to-trade ratio, cancel rate
- order lifetime mean/stddev (add → cancel/execute)
- layering score: longest run of same-side orders placed within a time
  window at non-decreasing distance from the touch
- cancel-burst z-score: how anomalously fast the latest cancel followed
  the previous one, relative to this participant's own history
- size-vs-baseline: latest order size vs. trailing mean, computed before
  the new size is folded into that mean

It's deliberately decoupled from `OrderBook`/LOBSTER replay: LOBSTER data
is anonymized (no real participant IDs), so callers supply `participant_id`
explicitly per event. Real per-participant tracking becomes meaningful once
synthetic ground-truth injection assigns participant IDs (see below).

## Ground truth

Public data has no labeled spoofing incidents. This project uses synthetic
injection as the primary evaluation methodology: clean LOBSTER data is treated
as non-manipulative, then synthetic layering/spoofing patterns are injected
with tracked ground-truth order IDs and timestamps, at graded difficulty
tiers. A secondary qualitative check compares detector behavior against
publicly documented cases (e.g., the Coscia spoofing prosecution) as a sanity
check, not a quantitative test.

`python/injection/` implements this: `participants.py` assigns every
background order a synthetic, Zipf-distributed participant_id (LOBSTER's
own order_id carries no participant identity); `patterns.py` generates
spoofing (one oversized order resting just behind the touch, pulled before
it can fill) and layering (a same-side order stack at non-decreasing
distance from touch — the exact shape `FeatureEngine`'s layering_score
looks for — cancelled in a burst) at three difficulty tiers (easy/medium/
hard, varying size multiplier, dwell time, and stack depth); `injector.py`
replays a LOBSTER message file in lockstep with LOBSTER's own orderbook
reference file so injected orders sit at real, non-crossing top-of-book
prices, then writes a participant-labeled augmented message stream and a
`ground_truth.csv` kept separate from the event stream itself:

```bash
python -m python.injection.injector \
  data/lobster_samples/AAPL_2012-06-21_10/message_10.csv \
  data/lobster_samples/AAPL_2012-06-21_10/orderbook_10.csv \
  data/synthetic/AAPL_2012-06-21 \
  --patterns-per-tier 15 --seed 0
```

## Model training

`spoofwatch_features` (`cpp/src/feature_dump.cpp`) replays an injected,
participant-labeled message file through the real `OrderBook` +
`FeatureEngine` — the same engine that will eventually sit in the hot
inference path — and writes one feature-vector row per order lifecycle
event:

```bash
./build/spoofwatch_features \
  data/synthetic/AAPL_2012-06-21/message_augmented.csv \
  data/synthetic/AAPL_2012-06-21/features.csv
```

`python/training/train.py` joins that against `ground_truth.csv`, keeping
only each order's feature snapshot at submission time (before its outcome
is known, so the label can't leak in), trains a `LightGBM` classifier on a
time-based split, and compares it against `python/eval/metrics.py`'s naive
cancel-rate-threshold baseline (which scores 0 recall — synthetic
manipulators are single-use identities with no prior cancel history to
threshold on):

```bash
python -m python.training.train \
  data/synthetic/AAPL_2012-06-21/features.csv \
  data/synthetic/AAPL_2012-06-21/ground_truth.csv
```

Precision/recall/F1 are reported at LightGBM's default 0.5 threshold,
which is a poor operating point given how imbalanced a single trading
day's injected-pattern rate is; PR-AUC is printed alongside as a more
honest ranking-quality number, and recall is broken out by difficulty
tier and pattern type rather than averaged away — it's uneven and
sample-size-sensitive run to run, which is expected with a few hundred
positives total, not hidden.

## Hot-path inference

`TreeModel` (`cpp/include/spoofwatch/tree_model.hpp`) evaluates a trained
LightGBM model inline, with no LightGBM or ONNX Runtime dependency in the
hot path and zero allocation in `predict_proba()`. `python/training/
export_model.py` flattens the booster's trees into a small binary format
(verified empirically: LightGBM's predicted probability is exactly
sigmoid(sum of each tree's leaf value), no hidden bias term to replicate).
`train.py --export-dir <dir>` writes both `model.bin` and a
`reference_predictions.csv` for cross-validation:

```bash
python -m python.training.train \
  data/synthetic/AAPL_2012-06-21/features.csv \
  data/synthetic/AAPL_2012-06-21/ground_truth.csv \
  --export-dir data/synthetic/AAPL_2012-06-21/export

./build/spoofwatch_infer_validate \
  data/synthetic/AAPL_2012-06-21/export/model.bin \
  data/synthetic/AAPL_2012-06-21/export/reference_predictions.csv
```

`predict_proba()` is branchless in the sense that matters for a real-time
engine: every tree is walked for a fixed number of iterations regardless
of its actual depth, and the next node is chosen with arithmetic
(`left + go_right * (right - left)`) instead of an if/else, so there's no
data-dependent branch on the split outcome to mispredict. Leaf nodes
self-loop (left == right == their own index, threshold == +inf), so a
path that reaches its leaf early just stays there for the remaining
iterations rather than needing a separate "have I hit a leaf yet?"
branch. On a real 120-pattern run (200 trees, 38,272 held-out rows), the
C++ evaluator's output matched Python's bit-for-bit — 0.0 max absolute
difference.

`spoofwatch_pipeline` (`cpp/src/pipeline.cpp`) assembles the full hot path
into one process — parse -> `OrderBook` -> `FeatureEngine` -> `TreeModel`
— scoring every `NEW` order in a single pass, rather than the separate
file-chained tools above:

```bash
./build/spoofwatch_pipeline \
  data/synthetic/AAPL_2012-06-21/message_augmented.csv \
  data/synthetic/AAPL_2012-06-21/export/model.bin \
  data/synthetic/AAPL_2012-06-21/scored.csv
```

This is Phase 6 groundwork — per-stage latency histograms and a
sustained-throughput benchmark against it are still to come (see
`docs/PHASES.md`).

## License

MIT
