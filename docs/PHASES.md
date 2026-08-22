# Phase Checklist

Checkpoint-gated, not calendar-gated — don't move to the next phase until the
current one's exit criteria pass. See README.md §Architecture for the full
design rationale.

- [x] **Phase 0 — Environment & Data**
  CMake + GoogleTest build, Python env, LOBSTER sample downloaded and parsed.
  Exit: can read a raw LOBSTER file and print structured events for a sample day.
  Done: `data/lobster_samples/AAPL_2012-06-21_10/` (400,391 events), parsed via
  `cpp/src/lobster_reader.cpp` / `spoofwatch <message.csv>`.

- [x] **Phase 1 — Order Book + Lifecycle Tracking**
  Fixed-size price-level book, order-ID → lifecycle hash map, pre-allocated pool.
  Exit: reconstructed book matches LOBSTER's reference output; unit tests pass.
  Done: `OrderPool` (open-addressing hash map + pre-allocated record pool,
  `cpp/src/order_pool.cpp`), `PriceLevelBook` (sorted fixed-capacity array,
  `cpp/src/price_level_book.cpp`), `OrderBook::apply()` handling all 6
  LOBSTER event types (`cpp/src/order_book.cpp`). 23/23 unit tests pass,
  including hand-computed toy sequences for every event type.
  Exit criterion redefined from "exact full-day match" to "verified against
  LOBSTER's reference with documented, investigated divergence": a
  depth-limited (10-level) LOBSTER message file cannot reconstruct the
  reference book exactly by design (see README §Order book reconstruction).
  Confirmed via cross-validation against an independent Python
  reimplementation (matching statistics to within 0.05%) and two
  byte-identical independent copies of the source data, ruling out a code
  bug or corrupted mirror.

- [x] **Phase 2 — Feature Engine**
  Ring-buffer rolling features per participant, incremental updates only.
  Exit: every feature unit-tested against hand-computed values on toy sequences.
  Done: `FeatureEngine` (`cpp/src/feature_engine.cpp`) tracks order-to-trade
  ratio, cancel rate, order-lifetime mean/stddev (Welford's online
  algorithm, `IncrementalStats`), layering score (longest run of same-side
  orders placed within a time window at non-decreasing distance from
  touch, via `RingBuffer`), cancel-burst z-score, and size-vs-baseline —
  all O(1) per event, zero heap allocation past construction. 16 new tests
  (39 total), including hand-computed toy sequences for every feature.
  Decoupled from OrderBook/LOBSTER replay on purpose: LOBSTER data is
  anonymized (no real participant IDs), so callers supply participant_id
  explicitly — real per-participant tracking becomes meaningful once
  Phase 3 injects synthetic participant IDs. Not yet wired into main.cpp.

- [x] **Phase 3 — Synthetic Ground Truth Generation**
  Layering/spoofing injection pipeline, difficulty tiers, ground truth tracked separately.
  Exit: labeled dataset generated with ground-truth IDs/timestamps, difficulty-tiered.
  Done: `python/injection/` — `participants.py` assigns every background
  LOBSTER order_id a synthetic, Zipf-distributed participant_id (real
  LOBSTER data is anonymized, so this is a prerequisite for per-participant
  features); `patterns.py` generates single-order spoofing patterns
  (large order resting one tick behind the touch, pulled before it can
  fill) and multi-order layering patterns (stack of same-side orders at
  non-decreasing distance from touch, matching `FeatureEngine`'s
  layering_score definition exactly, cancelled in a burst); `injector.py`
  reads a LOBSTER message file in lockstep with LOBSTER's own orderbook
  reference file (real top-of-book prices, so injected orders never cross
  the spread), injects patterns at spread-out points across three
  difficulty tiers (easy/medium/hard, tuned via size multiplier, dwell
  time, and order count), and writes an augmented, participant-labeled
  message stream plus a separate `ground_truth.csv` (pattern_id, type,
  tier, participant_id, order_ids, start/end time) — never mixed into the
  event stream itself. 5 pytest tests cover participant assignment
  determinism, pattern invariants (no injected execution rows, correct
  non-decreasing distance), and an end-to-end run. Verified on the full
  AAPL 2012-06-21 sample (400,391 background events, 45 injected patterns,
  15 per tier).

- [x] **Phase 4 — Model Training**
  LightGBM classifier, time-based split, precision/recall/F1/FPR.
  Exit: model beats naive cancel-rate-threshold baseline on held-out data.
  Done: `spoofwatch_features` (`cpp/src/feature_dump.cpp`) replays an
  injection-pipeline output file through the real `OrderBook` +
  `FeatureEngine` — not a Python reimplementation — and writes one
  feature-vector row per order lifecycle event, so training data and the
  eventual hot-path detector share one implementation. `python/training/
  dataset.py` joins those features against `ground_truth.csv`, keeping
  only each order's feature snapshot at submission time (its `NEW` event)
  so the label can't leak into the features — a real-time system only
  ever sees an order at the moment it arrives. `python/training/train.py`
  trains a `LGBMClassifier` on a time-based split (train on the earlier
  part of the day, test on the later part) and compares it against
  `python/eval/metrics.py`'s naive cancel-rate-threshold baseline, which
  scores 0 recall — synthetic manipulator participants are single-use
  identities with no prior history, so their historical cancel_rate at
  submission time is uninformative by construction. The model clears that
  bar comfortably (e.g. F1 0.46, PR-AUC ~130x the base positive rate on a
  120-pattern run), driven mostly by `order_to_trade_ratio`,
  `mean_lifetime_ns`, and `cancel_rate` rather than the layering-specific
  features. Recall is uneven and sample-size-sensitive across difficulty
  tier and pattern type run to run (single trading day, a few hundred
  positives total) — reported via `recall_by_group`, not smoothed over.
  Precision/recall/F1 are computed at LightGBM's default 0.5 threshold,
  which is a poor operating point under this much class imbalance;
  PR-AUC is printed alongside as the more honest ranking-quality number,
  and picking a real deployment threshold is left to Phase 7. 14 pytest
  tests cover the label join (including multi-order layering patterns)
  and metric functions against hand-computed values.

- [ ] **Phase 5 — Hot-Path Inference Integration**
  Branchless C++ tree evaluator or ONNX Runtime, zero-allocation event loop.
  Exit: C++ inference output matches Python model within float tolerance.

- [ ] **Phase 6 — Benchmarking & Profiling**
  Per-stage p50/p99/p99.9 latency, perf/flamegraphs, sustained throughput test.
  Exit: zero heap allocations observed in steady-state hot path.

- [ ] **Phase 7 — Evaluation & Writeup**
  Final precision/recall/FPR by difficulty tier, case-study sanity check, README.
  Exit: can walk the system end-to-end from memory and reproduce the numbers.

- [ ] **Phase 8 — Stretch (optional)**
  Multi-symbol support, unsupervised autoencoder comparison, SIMD feature computation.
