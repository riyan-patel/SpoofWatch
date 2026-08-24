#pragma once

#include <array>
#include <cstdint>

namespace spoofwatch {

// Fixed-size, log2-bucketed latency histogram — no heap allocation past
// construction, so it can be used to instrument the hot path it's
// measuring without itself becoming a source of the allocations Phase 6
// is checking for. Bucket k covers [2^k, 2^(k+1)) nanoseconds; with 48
// buckets that's up to ~78 hours, far more range than any single-event
// latency needs. Percentiles are approximate (bucket-boundary
// resolution, ~a factor of 2 within a bucket) — adequate for a p50/p99/
// p99.9 latency report, not for reproducing exact sample values.
class LatencyHistogram {
public:
    static constexpr int kNumBuckets = 48;

    void record(uint64_t ns);

    uint64_t count() const { return count_; }
    uint64_t min_ns() const { return count_ ? min_ns_ : 0; }
    uint64_t max_ns() const { return max_ns_; }
    double mean_ns() const { return count_ ? static_cast<double>(sum_ns_) / count_ : 0.0; }

    // p in [0, 100]. Returns the lower bound of the bucket containing the
    // p-th percentile sample.
    uint64_t percentile(double p) const;

    void reset();

private:
    static int bucket_index(uint64_t ns);

    std::array<uint64_t, kNumBuckets> buckets_{};
    uint64_t count_ = 0;
    uint64_t sum_ns_ = 0;
    uint64_t min_ns_ = UINT64_MAX;
    uint64_t max_ns_ = 0;
};

} // namespace spoofwatch
