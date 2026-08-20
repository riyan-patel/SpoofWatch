#include <gtest/gtest.h>

#include <cmath>

#include "spoofwatch/incremental_stats.hpp"

using spoofwatch::IncrementalStats;

TEST(IncrementalStats, EmptyIsZero) {
    IncrementalStats stats;
    EXPECT_EQ(stats.count, 0u);
    EXPECT_DOUBLE_EQ(stats.mean, 0.0);
    EXPECT_DOUBLE_EQ(stats.variance(), 0.0);
}

TEST(IncrementalStats, SingleValue) {
    IncrementalStats stats;
    stats.update(10.0);
    EXPECT_EQ(stats.count, 1u);
    EXPECT_DOUBLE_EQ(stats.mean, 10.0);
    EXPECT_DOUBLE_EQ(stats.variance(), 0.0); // undefined with n=1, defined as 0
}

TEST(IncrementalStats, MatchesHandComputedMeanAndVariance) {
    // Classic textbook example: mean=5, sum of squared deviations=32,
    // sample variance (n-1) = 32/7.
    IncrementalStats stats;
    for (double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) {
        stats.update(x);
    }
    EXPECT_EQ(stats.count, 8u);
    EXPECT_NEAR(stats.mean, 5.0, 1e-9);
    EXPECT_NEAR(stats.variance(), 32.0 / 7.0, 1e-9);
    EXPECT_NEAR(stats.stddev(), std::sqrt(32.0 / 7.0), 1e-9);
}
