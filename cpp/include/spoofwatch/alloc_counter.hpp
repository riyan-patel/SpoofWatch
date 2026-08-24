#pragma once

#include <cstdint>

namespace spoofwatch {

// Counts calls to the process-wide operator new/delete overrides defined
// in alloc_counter.cpp. Deliberately linked only into spoofwatch_benchmark
// (not spoofwatch_core), so overriding global allocation doesn't affect
// any other binary or the test suite — this exists purely to give Phase
// 6's "zero heap allocations in steady-state hot path" exit criterion an
// actual measurement instead of an assumption.
uint64_t alloc_count();
void reset_alloc_count();

} // namespace spoofwatch
