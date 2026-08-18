#include <gtest/gtest.h>

#include "spoofwatch/price_level_book.hpp"

using spoofwatch::PriceLevel;
using spoofwatch::PriceLevelBook;

TEST(PriceLevelBook, AddCreatesLevelAndAccumulates) {
    PriceLevelBook book(16, /*ascending=*/true);
    book.add_qty(100, 10);
    book.add_qty(100, 5);
    ASSERT_EQ(book.level_count(), 1u);
    EXPECT_EQ(book.level_at(0).price, 100);
    EXPECT_EQ(book.level_at(0).size, 15u);
}

TEST(PriceLevelBook, AscendingKeepsBestAskFirst) {
    PriceLevelBook asks(16, /*ascending=*/true);
    asks.add_qty(300, 1);
    asks.add_qty(100, 1);
    asks.add_qty(200, 1);
    ASSERT_EQ(asks.level_count(), 3u);
    EXPECT_EQ(asks.level_at(0).price, 100);
    EXPECT_EQ(asks.level_at(1).price, 200);
    EXPECT_EQ(asks.level_at(2).price, 300);
}

TEST(PriceLevelBook, DescendingKeepsBestBidFirst) {
    PriceLevelBook bids(16, /*ascending=*/false);
    bids.add_qty(100, 1);
    bids.add_qty(300, 1);
    bids.add_qty(200, 1);
    ASSERT_EQ(bids.level_count(), 3u);
    EXPECT_EQ(bids.level_at(0).price, 300);
    EXPECT_EQ(bids.level_at(1).price, 200);
    EXPECT_EQ(bids.level_at(2).price, 100);
}

TEST(PriceLevelBook, RemoveQtyDeletesEmptyLevel) {
    PriceLevelBook book(16, true);
    book.add_qty(100, 10);
    EXPECT_TRUE(book.remove_qty(100, 10));
    EXPECT_EQ(book.level_count(), 0u);
}

TEST(PriceLevelBook, RemoveQtyPartialLeavesLevel) {
    PriceLevelBook book(16, true);
    book.add_qty(100, 10);
    EXPECT_TRUE(book.remove_qty(100, 4));
    ASSERT_EQ(book.level_count(), 1u);
    EXPECT_EQ(book.level_at(0).size, 6u);
}

TEST(PriceLevelBook, RemoveQtyFailsOnMissingOrInsufficient) {
    PriceLevelBook book(16, true);
    book.add_qty(100, 5);
    EXPECT_FALSE(book.remove_qty(200, 1)); // no such level
    EXPECT_FALSE(book.remove_qty(100, 6)); // more than resting
}

TEST(PriceLevelBook, TopNPadsWithDummySentinel) {
    PriceLevelBook asks(16, /*ascending=*/true);
    asks.add_qty(100, 5);

    std::vector<PriceLevel> out;
    asks.top_n(3, out);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0].price, 100);
    EXPECT_EQ(out[0].size, 5u);
    EXPECT_EQ(out[1].price, 9999999999LL);
    EXPECT_EQ(out[1].size, 0u);
    EXPECT_EQ(out[2].price, 9999999999LL);

    PriceLevelBook bids(16, /*ascending=*/false);
    std::vector<PriceLevel> out_bids;
    bids.top_n(2, out_bids);
    ASSERT_EQ(out_bids.size(), 2u);
    EXPECT_EQ(out_bids[0].price, -9999999999LL);
}
