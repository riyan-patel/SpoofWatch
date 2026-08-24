#include "spoofwatch/latency_histogram.hpp"

#include <algorithm>

namespace spoofwatch {

int LatencyHistogram::bucket_index(uint64_t ns) {
    if (ns == 0) {
        return 0;
    }
    // 63 - count_leading_zeros(ns) == floor(log2(ns)).
    int bit = 63;
    while (bit >= 0 && !((ns >> bit) & 1u)) {
        --bit;
    }
    return std::min(bit, kNumBuckets - 1);
}

void LatencyHistogram::record(uint64_t ns) {
    ++buckets_[bucket_index(ns)];
    ++count_;
    sum_ns_ += ns;
    min_ns_ = std::min(min_ns_, ns);
    max_ns_ = std::max(max_ns_, ns);
}

uint64_t LatencyHistogram::percentile(double p) const {
    if (count_ == 0) {
        return 0;
    }
    uint64_t target = static_cast<uint64_t>(p / 100.0 * static_cast<double>(count_));
    if (target > 0) {
        --target; // convert "target-th sample" to a 0-indexed cumulative threshold
    }
    uint64_t cumulative = 0;
    for (int i = 0; i < kNumBuckets; ++i) {
        cumulative += buckets_[i];
        if (cumulative > target) {
            return static_cast<uint64_t>(1) << i;
        }
    }
    return max_ns_;
}

void LatencyHistogram::reset() {
    buckets_.fill(0);
    count_ = 0;
    sum_ns_ = 0;
    min_ns_ = UINT64_MAX;
    max_ns_ = 0;
}

} // namespace spoofwatch
