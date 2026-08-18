#include <gtest/gtest.h>

#include "spoofwatch/order_book.hpp"

using spoofwatch::LobsterEventType;
using spoofwatch::LobsterMessage;
using spoofwatch::OrderBook;
using spoofwatch::PriceLevel;

namespace {

LobsterMessage msg(double t, LobsterEventType type, uint64_t id, uint32_t size, int64_t price, int32_t dir) {
    return LobsterMessage{t, type, id, size, price, dir};
}

} // namespace

TEST(OrderBook, NewOrdersPopulateBothSides) {
    OrderBook book(64, 64);
    book.apply(msg(1.0, LobsterEventType::NewLimitOrder, 1, 10, 1000000, 1));  // buy $100 x10
    book.apply(msg(1.1, LobsterEventType::NewLimitOrder, 2, 5, 1010000, -1)); // sell $101 x5

    std::vector<PriceLevel> asks, bids;
    book.top_n(1, asks, bids);
    EXPECT_EQ(bids[0].price, 1000000);
    EXPECT_EQ(bids[0].size, 10u);
    EXPECT_EQ(asks[0].price, 1010000);
    EXPECT_EQ(asks[0].size, 5u);
}

TEST(OrderBook, PartialCancelReducesLevelAndOrderSize) {
    OrderBook book(64, 64);
    book.apply(msg(1.0, LobsterEventType::NewLimitOrder, 1, 10, 1000000, 1));
    book.apply(msg(1.1, LobsterEventType::PartialCancel, 1, 3, 1000000, 1));

    std::vector<PriceLevel> asks, bids;
    book.top_n(1, asks, bids);
    EXPECT_EQ(bids[0].size, 7u);
    EXPECT_EQ(book.pool().find(1)->size, 7u);
}

TEST(OrderBook, VisibleExecutionRemovesFullyFilledOrder) {
    OrderBook book(64, 64);
    book.apply(msg(1.0, LobsterEventType::NewLimitOrder, 2, 5, 1010000, -1));
    book.apply(msg(1.1, LobsterEventType::VisibleExecution, 2, 5, 1010000, -1));

    std::vector<PriceLevel> asks, bids;
    book.top_n(1, asks, bids);
    EXPECT_EQ(asks[0].size, 0u); // level gone, padded with dummy
    EXPECT_EQ(book.pool().find(2), nullptr);
}

TEST(OrderBook, DeletionRemovesRemainingSizeAndOrder) {
    OrderBook book(64, 64);
    book.apply(msg(1.0, LobsterEventType::NewLimitOrder, 1, 10, 1000000, 1));
    book.apply(msg(1.1, LobsterEventType::PartialCancel, 1, 3, 1000000, 1)); // now size 7
    book.apply(msg(1.2, LobsterEventType::Deletion, 1, 7, 1000000, 1));

    std::vector<PriceLevel> asks, bids;
    book.top_n(1, asks, bids);
    EXPECT_EQ(bids[0].size, 0u);
    EXPECT_EQ(book.pool().find(1), nullptr);
}

TEST(OrderBook, HiddenExecutionAndTradingHaltAreNoOps) {
    OrderBook book(64, 64);
    book.apply(msg(1.0, LobsterEventType::NewLimitOrder, 1, 10, 1000000, 1));
    book.apply(msg(1.1, LobsterEventType::HiddenExecution, 999, 1, 1000000, 1));
    book.apply(msg(1.2, LobsterEventType::TradingHalt, 0, 0, -1, -1));

    std::vector<PriceLevel> asks, bids;
    book.top_n(1, asks, bids);
    EXPECT_EQ(bids[0].size, 10u); // untouched
}

TEST(OrderBook, MultipleOrdersAtSamePriceAggregate) {
    OrderBook book(64, 64);
    book.apply(msg(1.0, LobsterEventType::NewLimitOrder, 1, 10, 1000000, 1));
    book.apply(msg(1.1, LobsterEventType::NewLimitOrder, 2, 20, 1000000, 1));

    std::vector<PriceLevel> asks, bids;
    book.top_n(1, asks, bids);
    EXPECT_EQ(bids[0].size, 30u);

    book.apply(msg(1.2, LobsterEventType::Deletion, 1, 10, 1000000, 1));
    book.top_n(1, asks, bids);
    EXPECT_EQ(bids[0].size, 20u); // order 2's quantity remains
}
