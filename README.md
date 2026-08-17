# SpoofWatch

Low-latency C++ order-flow surveillance engine with inline ML-based spoofing/layering detection.

**Status: scaffold only.** See [docs/PHASES.md](docs/PHASES.md) for the build plan — nothing past Phase 0 is implemented yet.

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
                    [4] ML Inference (branchless GBT / ONNX)
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
| Hot-path inference | Hand-rolled branchless tree evaluator or ONNX Runtime C++ API |
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
```

## Ground truth

Public data has no labeled spoofing incidents. This project uses synthetic
injection as the primary evaluation methodology: clean LOBSTER data is treated
as non-manipulative, then synthetic layering/spoofing patterns are injected
with tracked ground-truth order IDs and timestamps, at graded difficulty
tiers. A secondary qualitative check compares detector behavior against
publicly documented cases (e.g., the Coscia spoofing prosecution) as a sanity
check, not a quantitative test. Full rationale in [docs/PHASES.md](docs/PHASES.md).

## License

MIT
