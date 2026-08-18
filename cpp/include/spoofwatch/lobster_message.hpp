#pragma once

#include <cstdint>

namespace spoofwatch {

// Mirrors LOBSTER's message file event types exactly.
// See data/lobster_samples/*/README.txt for the format spec.
enum class LobsterEventType : int32_t {
    NewLimitOrder = 1,
    PartialCancel = 2,
    Deletion = 3,
    VisibleExecution = 4,
    HiddenExecution = 5,
    TradingHalt = 7,
};

// One row of a LOBSTER message file, parsed but not yet interpreted into
// an OrderRecord/book update. Field types match the column semantics in
// the README (price is dollars * 10000, direction is +1 buy / -1 sell).
struct LobsterMessage {
    double time_sec;        // seconds after midnight, sub-ms precision
    LobsterEventType type;
    uint64_t order_id;
    uint32_t size;
    int64_t price;          // dollar price * 10000
    int32_t direction;      // +1 = buy limit order, -1 = sell limit order
};

} // namespace spoofwatch
