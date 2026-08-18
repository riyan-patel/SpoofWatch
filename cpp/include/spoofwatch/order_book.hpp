#pragma once

#include <cstddef>
#include <vector>

#include "spoofwatch/lobster_message.hpp"
#include "spoofwatch/order_pool.hpp"
#include "spoofwatch/price_level_book.hpp"

namespace spoofwatch {

// Reconstructs a single symbol's limit order book from a LOBSTER message
// stream. All storage (order pool, price levels) is pre-allocated at
// construction — apply() never allocates on the heap.
class OrderBook {
public:
    OrderBook(size_t max_orders, size_t max_price_levels);

    // Applies one message to the book, updating price levels and the
    // order lifecycle pool. Trading-halt messages (type 7) and hidden
    // executions (type 5, which reference orders never added to the
    // visible book) are no-ops.
    void apply(const LobsterMessage& msg);

    // Fills `asks`/`bids` with the top n price levels, LOBSTER-orderbook-
    // row format: best level first, padded with the dummy sentinel if the
    // book is shallower than n.
    void top_n(size_t n, std::vector<PriceLevel>& asks, std::vector<PriceLevel>& bids) const;

    const OrderPool& pool() const { return pool_; }

private:
    OrderPool pool_;
    PriceLevelBook bids_; // sorted descending: best bid first
    PriceLevelBook asks_; // sorted ascending: best ask first
};

} // namespace spoofwatch
