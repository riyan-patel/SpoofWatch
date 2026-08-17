#pragma once

#include <cstdint>

namespace spoofwatch {

enum class OrderStatus : uint8_t {
    Active,
    Cancelled,
    Executed,
    Partial,
};

enum class Side : uint8_t {
    Bid = 0,
    Ask = 1,
};

// Fixed-size record stored in a pre-allocated pool. No pointers, no heap
// allocation — this struct is the unit the whole hot path is built around.
struct OrderRecord {
    uint64_t order_id;
    uint64_t participant_id;
    uint64_t ts_added_ns;
    uint64_t ts_last_event_ns;
    double price;
    uint32_t size;
    Side side;
    OrderStatus status;
};

} // namespace spoofwatch
