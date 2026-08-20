#include <gtest/gtest.h>

#include "spoofwatch/feature_engine.hpp"

using spoofwatch::FeatureEngine;
using spoofwatch::Side;

TEST(FeatureEngine, UnknownParticipantReturnsZeroedSnapshot) {
    FeatureEngine engine(16, 16);
    auto snap = engine.snapshot(999);
    EXPECT_EQ(snap.order_to_trade_ratio, 0.0);
    EXPECT_EQ(snap.layering_score_bid, 0u);
}

TEST(FeatureEngine, OrderToTradeRatio) {
    FeatureEngine engine(16, 16);
    engine.on_new_order(1, 101, Side::Bid, 10, 0, 1000);
    engine.on_new_order(1, 102, Side::Bid, 10, 0, 1100);
    engine.on_new_order(1, 103, Side::Bid, 10, 0, 1200);
    engine.on_execute(1, 101, 1300); // one of the three trades

    auto snap = engine.snapshot(1);
    EXPECT_DOUBLE_EQ(snap.order_to_trade_ratio, 3.0 / 1.0);
}

TEST(FeatureEngine, CancelRate) {
    FeatureEngine engine(16, 16);
    for (uint64_t id = 1; id <= 4; ++id) {
        engine.on_new_order(1, id, Side::Ask, 10, 0, id * 100);
    }
    engine.on_cancel(1, 1, 500);
    engine.on_cancel(1, 2, 600);

    auto snap = engine.snapshot(1);
    EXPECT_DOUBLE_EQ(snap.cancel_rate, 2.0 / 4.0);
}

TEST(FeatureEngine, LifetimeStatsMatchHandComputedMean) {
    FeatureEngine engine(16, 16);
    engine.on_new_order(1, 1, Side::Bid, 10, 0, 1000);
    engine.on_cancel(1, 1, 5000); // lifetime 4000

    engine.on_new_order(1, 2, Side::Bid, 10, 0, 2000);
    engine.on_cancel(1, 2, 6000); // lifetime 4000

    engine.on_new_order(1, 3, Side::Bid, 10, 0, 3000);
    engine.on_cancel(1, 3, 9000); // lifetime 6000

    // mean of [4000, 4000, 6000] = 4666.667
    auto snap = engine.snapshot(1);
    EXPECT_NEAR(snap.mean_lifetime_ns, 14000.0 / 3.0, 1e-6);
}

TEST(FeatureEngine, LayeringScoreCountsIncreasingDistanceWithinWindow) {
    FeatureEngine engine(16, 16, /*layering_window_ns=*/500'000'000);
    // 4 same-side orders, increasing distance from touch, tightly spaced in time.
    engine.on_new_order(1, 1, Side::Ask, 10, /*distance=*/1, 0);
    engine.on_new_order(1, 2, Side::Ask, 10, /*distance=*/2, 100);
    engine.on_new_order(1, 3, Side::Ask, 10, /*distance=*/3, 200);
    engine.on_new_order(1, 4, Side::Ask, 10, /*distance=*/4, 300);

    EXPECT_EQ(engine.snapshot(1).layering_score_ask, 4u);

    // A closer order breaks the increasing-distance run.
    engine.on_new_order(1, 5, Side::Ask, 10, /*distance=*/0, 400);
    EXPECT_EQ(engine.snapshot(1).layering_score_ask, 1u);
}

TEST(FeatureEngine, LayeringScoreResetsOutsideTimeWindow) {
    FeatureEngine engine(16, 16, /*layering_window_ns=*/100); // tiny window
    engine.on_new_order(1, 1, Side::Bid, 10, 1, 0);
    engine.on_new_order(1, 2, Side::Bid, 10, 2, 10);   // within window (gap=10)
    engine.on_new_order(1, 3, Side::Bid, 10, 3, 5000); // far outside window

    EXPECT_EQ(engine.snapshot(1).layering_score_bid, 1u);
}

TEST(FeatureEngine, CancelBurstZScoreMatchesHandComputedValue) {
    FeatureEngine engine(16, 16);
    engine.on_new_order(1, 1, Side::Bid, 10, 0, 0);
    engine.on_new_order(1, 2, Side::Bid, 10, 0, 0);
    engine.on_new_order(1, 3, Side::Bid, 10, 0, 0);
    engine.on_new_order(1, 4, Side::Bid, 10, 0, 0);

    engine.on_cancel(1, 1, 0);     // first cancel: no gap yet
    engine.on_cancel(1, 2, 1000);  // gap=1000, first sample recorded after this
    engine.on_cancel(1, 3, 3000);  // gap=2000; now cancel_gap_ns has [1000], count=1 -> z=0 default
    // After this cancel, cancel_gap_ns has [1000, 2000]: mean=1500, sample stddev=sqrt(500000)=707.1067811865476
    engine.on_cancel(1, 4, 3100);  // gap=100; z = (1500 - 100) / 707.1067811865476

    EXPECT_NEAR(engine.snapshot(1).cancel_burst_zscore, 1400.0 / 707.1067811865476, 1e-6);
}

TEST(FeatureEngine, SizeVsBaselineComparesAgainstPriorMeanOnly) {
    FeatureEngine engine(16, 16);
    engine.on_new_order(1, 1, Side::Bid, 100, 0, 0);
    engine.on_new_order(1, 2, Side::Bid, 100, 0, 0);
    engine.on_new_order(1, 3, Side::Bid, 100, 0, 0);
    // baseline mean from first 3 orders = 100
    engine.on_new_order(1, 4, Side::Bid, 300, 0, 0);

    EXPECT_DOUBLE_EQ(engine.snapshot(1).size_vs_baseline_ratio, 3.0);
}

TEST(FeatureEngine, ParticipantsAreIsolated) {
    FeatureEngine engine(16, 16);
    engine.on_new_order(1, 1, Side::Bid, 10, 0, 0);
    engine.on_new_order(1, 2, Side::Bid, 10, 0, 0);
    engine.on_new_order(2, 3, Side::Bid, 10, 0, 0);
    engine.on_execute(2, 3, 100);

    EXPECT_DOUBLE_EQ(engine.snapshot(1).order_to_trade_ratio, 2.0); // placed=2, executed=0 -> /1
    EXPECT_DOUBLE_EQ(engine.snapshot(2).order_to_trade_ratio, 1.0); // placed=1, executed=1
}
