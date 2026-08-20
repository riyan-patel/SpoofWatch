#pragma once

#include <cstdint>
#include <vector>

#include "spoofwatch/incremental_stats.hpp"
#include "spoofwatch/order_record.hpp"
#include "spoofwatch/ring_buffer.hpp"

namespace spoofwatch {

struct ParticipantFeatureSnapshot {
    double order_to_trade_ratio = 0.0;   // orders_placed / max(orders_executed, 1)
    double cancel_rate = 0.0;            // orders_cancelled / max(orders_placed, 1)
    double mean_lifetime_ns = 0.0;       // Welford mean of (close_ts - add_ts)
    double lifetime_stddev_ns = 0.0;
    uint32_t layering_score_bid = 0;     // longest run of same-side orders placed within
    uint32_t layering_score_ask = 0;     // the layering window at non-decreasing distance from touch
    double cancel_burst_zscore = 0.0;    // vs. this participant's historical inter-cancel gap
    double size_vs_baseline_ratio = 0.0; // most recent order size / trailing mean size
};

// Tracks per-participant rolling features, updated incrementally on each
// order lifecycle event — never recomputed from full history. All storage
// is pre-allocated at construction.
//
// Decoupled from OrderBook/LOBSTER replay on purpose: LOBSTER data is
// anonymized (no real participant IDs), so real per-participant tracking
// only becomes meaningful once Phase 3 injects synthetic participant IDs.
// Callers supply participant_id explicitly per event.
class FeatureEngine {
public:
    FeatureEngine(size_t max_participants, size_t max_tracked_orders,
                  uint64_t layering_window_ns = 500'000'000);

    // distance_from_touch: this order's price distance from the opposite
    // side's best price at insertion time, in whatever units the caller
    // uses consistently (raw price, ticks). Larger = farther from touch.
    void on_new_order(uint64_t participant_id, uint64_t order_id, Side side, uint32_t size,
                       int64_t distance_from_touch, uint64_t ts_ns);
    void on_cancel(uint64_t participant_id, uint64_t order_id, uint64_t ts_ns);
    void on_execute(uint64_t participant_id, uint64_t order_id, uint64_t ts_ns);

    // Returns a zeroed snapshot if participant_id has never been seen.
    ParticipantFeatureSnapshot snapshot(uint64_t participant_id) const;

private:
    struct ActiveOrderInfo {
        int64_t distance = 0;
        uint64_t ts_ns = 0;
    };

    static constexpr size_t kLayeringCapacity = 8;

    struct ParticipantState {
        uint64_t participant_id = 0;
        uint64_t orders_placed = 0;
        uint64_t orders_executed = 0;
        uint64_t orders_cancelled = 0;
        IncrementalStats lifetime_ns;
        IncrementalStats size_stats;
        IncrementalStats cancel_gap_ns;
        uint64_t last_cancel_ts_ns = 0;
        bool has_last_cancel = false;
        double last_cancel_burst_zscore = 0.0;
        double last_size_vs_baseline_ratio = 0.0;
        RingBuffer<ActiveOrderInfo, kLayeringCapacity> bid_recent;
        RingBuffer<ActiveOrderInfo, kLayeringCapacity> ask_recent;
        uint32_t layering_score_bid = 0;
        uint32_t layering_score_ask = 0;
    };

    struct OrderMeta {
        uint64_t participant_id = 0;
        uint64_t ts_added_ns = 0;
    };

    // Both maps below use the same fixed-capacity open-addressing scheme
    // as OrderPool (see order_pool.hpp) — separate implementations here
    // since the payloads differ and Phase 1's OrderPool is already in use
    // by OrderBook for an unrelated purpose (price-level tracking).
    enum class SlotState : uint8_t { Empty, Occupied, Tombstone };

    struct ParticipantSlot {
        uint64_t key = 0;
        uint32_t index = 0;
        SlotState state = SlotState::Empty;
    };
    struct OrderSlot {
        uint64_t key = 0;
        OrderMeta meta;
        SlotState state = SlotState::Empty;
    };

    ParticipantState& get_or_create_participant(uint64_t participant_id);
    ParticipantState* find_participant(uint64_t participant_id);
    const ParticipantState* find_participant(uint64_t participant_id) const;

    void insert_order_meta(uint64_t order_id, OrderMeta meta);
    const OrderMeta* find_order_meta(uint64_t order_id) const;
    void erase_order_meta(uint64_t order_id);

    static uint32_t layering_score(const RingBuffer<ActiveOrderInfo, kLayeringCapacity>& buf,
                                    uint64_t window_ns);

    uint64_t layering_window_ns_;

    std::vector<ParticipantState> participants_;
    std::vector<uint32_t> participant_free_list_;
    size_t participant_free_top_;
    std::vector<ParticipantSlot> participant_hash_;
    size_t participant_hash_mask_;

    std::vector<OrderSlot> order_hash_;
    size_t order_hash_mask_;
};

} // namespace spoofwatch
