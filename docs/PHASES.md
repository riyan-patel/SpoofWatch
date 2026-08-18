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

- [ ] **Phase 2 — Feature Engine**
  Ring-buffer rolling features per participant, incremental updates only.
  Exit: every feature unit-tested against hand-computed values on toy sequences.

- [ ] **Phase 3 — Synthetic Ground Truth Generation**
  Layering/spoofing injection pipeline, difficulty tiers, ground truth tracked separately.
  Exit: labeled dataset generated with ground-truth IDs/timestamps, difficulty-tiered.

- [ ] **Phase 4 — Model Training**
  LightGBM classifier, time-based split, precision/recall/F1/FPR.
  Exit: model beats naive cancel-rate-threshold baseline on held-out data.

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
