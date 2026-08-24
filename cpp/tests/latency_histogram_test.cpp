#include <gtest/gtest.h>

#include "spoofwatch/latency_histogram.hpp"

using spoofwatch::LatencyHistogram;

TEST(LatencyHistogram, EmptyIsZero) {
    LatencyHistogram h;
    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.min_ns(), 0u);
    EXPECT_EQ(h.max_ns(), 0u);
    EXPECT_DOUBLE_EQ(h.mean_ns(), 0.0);
    EXPECT_EQ(h.percentile(50), 0u);
}

TEST(LatencyHistogram, SingleValueReportsItself) {
    LatencyHistogram h;
    h.record(1000);
    EXPECT_EQ(h.count(), 1u);
    EXPECT_EQ(h.min_ns(), 1000u);
    EXPECT_EQ(h.max_ns(), 1000u);
    EXPECT_DOUBLE_EQ(h.mean_ns(), 1000.0);
    // 1000 falls in bucket floor(log2(1000))=9 -> [512, 1024)
    EXPECT_EQ(h.percentile(50), 512u);
}

TEST(LatencyHistogram, MeanMinMaxMatchHandComputedValues) {
    LatencyHistogram h;
    for (uint64_t v : {10u, 20u, 30u, 40u}) {
        h.record(v);
    }
    EXPECT_EQ(h.count(), 4u);
    EXPECT_EQ(h.min_ns(), 10u);
    EXPECT_EQ(h.max_ns(), 40u);
    EXPECT_DOUBLE_EQ(h.mean_ns(), 25.0);
}

TEST(LatencyHistogram, PercentilesAreMonotonicallyNonDecreasing) {
    LatencyHistogram h;
    for (uint64_t v = 1; v <= 1000; ++v) {
        h.record(v * 1000);
    }
    uint64_t p50 = h.percentile(50);
    uint64_t p99 = h.percentile(99);
    uint64_t p999 = h.percentile(99.9);
    uint64_t p100 = h.percentile(100);
    EXPECT_LE(p50, p99);
    EXPECT_LE(p99, p999);
    EXPECT_LE(p999, p100);
    // p50 of a uniform 1..1000 (in units of 1000ns) distribution should be
    // in the neighborhood of 500_000ns, within this histogram's bucket
    // resolution (a factor of ~2).
    EXPECT_GE(p50, 200000u);
    EXPECT_LE(p50, 700000u);
}

TEST(LatencyHistogram, ZeroIsHandledWithoutUnderflow) {
    LatencyHistogram h;
    h.record(0);
    h.record(0);
    EXPECT_EQ(h.count(), 2u);
    EXPECT_EQ(h.min_ns(), 0u);
    EXPECT_EQ(h.max_ns(), 0u);
}

TEST(LatencyHistogram, ResetClearsAllState) {
    LatencyHistogram h;
    h.record(500);
    h.record(999999);
    h.reset();
    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.min_ns(), 0u);
    EXPECT_EQ(h.max_ns(), 0u);
    EXPECT_DOUBLE_EQ(h.mean_ns(), 0.0);
}

TEST(LatencyHistogram, VeryLargeValueDoesNotOverflowBucketIndex) {
    LatencyHistogram h;
    h.record(UINT64_MAX);
    EXPECT_EQ(h.count(), 1u);
    EXPECT_EQ(h.max_ns(), UINT64_MAX);
    // Clamped into the top bucket rather than indexing out of range.
    EXPECT_EQ(h.percentile(100), static_cast<uint64_t>(1) << (LatencyHistogram::kNumBuckets - 1));
}
