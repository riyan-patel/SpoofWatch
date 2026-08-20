#pragma once

#include <cmath>
#include <cstdint>

namespace spoofwatch {

// Welford's online algorithm: mean and variance updated in O(1) per
// sample, no history retained — the "never recomputed from scratch"
// discipline the feature engine is built around.
struct IncrementalStats {
    uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void update(double x) {
        ++count;
        double delta = x - mean;
        mean += delta / static_cast<double>(count);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance() const {
        return count > 1 ? m2 / static_cast<double>(count - 1) : 0.0;
    }

    double stddev() const { return std::sqrt(variance()); }
};

} // namespace spoofwatch
