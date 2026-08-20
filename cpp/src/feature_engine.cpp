#include "spoofwatch/feature_engine.hpp"

#include <algorithm>
#include <stdexcept>

namespace spoofwatch {

namespace {

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

uint64_t hash_u64(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

} // namespace

FeatureEngine::FeatureEngine(size_t max_participants, size_t max_tracked_orders, uint64_t layering_window_ns)
    : layering_window_ns_(layering_window_ns),
      participants_(max_participants) {
    // participant storage: pre-allocated array + free list + open-addressing index
    participant_free_list_.resize(max_participants);
    for (size_t i = 0; i < max_participants; ++i) {
        participant_free_list_[i] = static_cast<uint32_t>(max_participants - 1 - i);
    }
    participant_free_top_ = max_participants;

    size_t participant_hash_capacity = next_power_of_two(std::max<size_t>(max_participants * 2, 2));
    participant_hash_.resize(participant_hash_capacity);
    participant_hash_mask_ = participant_hash_capacity - 1;

    size_t order_hash_capacity = next_power_of_two(std::max<size_t>(max_tracked_orders * 2, 2));
    order_hash_.resize(order_hash_capacity);
    order_hash_mask_ = order_hash_capacity - 1;
}

FeatureEngine::ParticipantState& FeatureEngine::get_or_create_participant(uint64_t participant_id) {
    if (ParticipantState* existing = find_participant(participant_id)) {
        return *existing;
    }

    size_t idx = hash_u64(participant_id) & participant_hash_mask_;
    for (size_t probes = 0; probes < participant_hash_.size(); ++probes) {
        if (participant_hash_[idx].state == SlotState::Empty) {
            if (participant_free_top_ == 0) {
                throw std::runtime_error("FeatureEngine: participant capacity exceeded");
            }
            uint32_t slot = participant_free_list_[--participant_free_top_];
            participant_hash_[idx] = ParticipantSlot{participant_id, slot, SlotState::Occupied};
            participants_[slot] = ParticipantState{};
            participants_[slot].participant_id = participant_id;
            return participants_[slot];
        }
        idx = (idx + 1) & participant_hash_mask_;
    }
    throw std::runtime_error("FeatureEngine: participant hash table full");
}

FeatureEngine::ParticipantState* FeatureEngine::find_participant(uint64_t participant_id) {
    size_t idx = hash_u64(participant_id) & participant_hash_mask_;
    for (size_t probes = 0; probes < participant_hash_.size(); ++probes) {
        const ParticipantSlot& slot = participant_hash_[idx];
        if (slot.state == SlotState::Empty) {
            return nullptr;
        }
        if (slot.state == SlotState::Occupied && slot.key == participant_id) {
            return &participants_[slot.index];
        }
        idx = (idx + 1) & participant_hash_mask_;
    }
    return nullptr;
}

const FeatureEngine::ParticipantState* FeatureEngine::find_participant(uint64_t participant_id) const {
    return const_cast<FeatureEngine*>(this)->find_participant(participant_id);
}

void FeatureEngine::insert_order_meta(uint64_t order_id, OrderMeta meta) {
    size_t idx = hash_u64(order_id) & order_hash_mask_;
    size_t first_tombstone = order_hash_.size();
    for (size_t probes = 0; probes < order_hash_.size(); ++probes) {
        OrderSlot& slot = order_hash_[idx];
        if (slot.state == SlotState::Occupied && slot.key == order_id) {
            slot.meta = meta; // overwrite (shouldn't normally happen for valid data)
            return;
        }
        if (slot.state == SlotState::Tombstone && first_tombstone == order_hash_.size()) {
            first_tombstone = idx;
        }
        if (slot.state == SlotState::Empty) {
            size_t target = (first_tombstone != order_hash_.size()) ? first_tombstone : idx;
            order_hash_[target] = OrderSlot{order_id, meta, SlotState::Occupied};
            return;
        }
        idx = (idx + 1) & order_hash_mask_;
    }
    // Table full — drop silently. Lifetime/layering tracking degrades
    // gracefully rather than crashing the hot path.
}

const FeatureEngine::OrderMeta* FeatureEngine::find_order_meta(uint64_t order_id) const {
    size_t idx = hash_u64(order_id) & order_hash_mask_;
    for (size_t probes = 0; probes < order_hash_.size(); ++probes) {
        const OrderSlot& slot = order_hash_[idx];
        if (slot.state == SlotState::Empty) {
            return nullptr;
        }
        if (slot.state == SlotState::Occupied && slot.key == order_id) {
            return &slot.meta;
        }
        idx = (idx + 1) & order_hash_mask_;
    }
    return nullptr;
}

void FeatureEngine::erase_order_meta(uint64_t order_id) {
    size_t idx = hash_u64(order_id) & order_hash_mask_;
    for (size_t probes = 0; probes < order_hash_.size(); ++probes) {
        OrderSlot& slot = order_hash_[idx];
        if (slot.state == SlotState::Empty) {
            return;
        }
        if (slot.state == SlotState::Occupied && slot.key == order_id) {
            slot.state = SlotState::Tombstone;
            return;
        }
        idx = (idx + 1) & order_hash_mask_;
    }
}

uint32_t FeatureEngine::layering_score(const RingBuffer<ActiveOrderInfo, kLayeringCapacity>& buf,
                                        uint64_t window_ns) {
    size_t n = buf.size();
    if (n == 0) {
        return 0;
    }
    uint32_t run = 1;
    for (size_t i = n - 1; i > 0; --i) {
        const ActiveOrderInfo& newer = buf[i];
        const ActiveOrderInfo& older = buf[i - 1];
        if ((newer.ts_ns - older.ts_ns) <= window_ns && older.distance <= newer.distance) {
            ++run;
        } else {
            break;
        }
    }
    return run;
}

void FeatureEngine::on_new_order(uint64_t participant_id, uint64_t order_id, Side side, uint32_t size,
                                  int64_t distance_from_touch, uint64_t ts_ns) {
    ParticipantState& p = get_or_create_participant(participant_id);

    ++p.orders_placed;

    p.last_size_vs_baseline_ratio = (p.size_stats.count > 0 && p.size_stats.mean > 0.0)
                                         ? static_cast<double>(size) / p.size_stats.mean
                                         : 1.0;
    p.size_stats.update(static_cast<double>(size));

    auto& buf = (side == Side::Bid) ? p.bid_recent : p.ask_recent;
    buf.push_back(ActiveOrderInfo{distance_from_touch, ts_ns});
    uint32_t score = layering_score(buf, layering_window_ns_);
    if (side == Side::Bid) {
        p.layering_score_bid = score;
    } else {
        p.layering_score_ask = score;
    }

    insert_order_meta(order_id, OrderMeta{participant_id, ts_ns});
}

void FeatureEngine::on_cancel(uint64_t participant_id, uint64_t order_id, uint64_t ts_ns) {
    ParticipantState& p = get_or_create_participant(participant_id);
    ++p.orders_cancelled;

    if (p.has_last_cancel) {
        uint64_t gap = ts_ns - p.last_cancel_ts_ns;
        if (p.cancel_gap_ns.count >= 2 && p.cancel_gap_ns.stddev() > 0.0) {
            p.last_cancel_burst_zscore = (p.cancel_gap_ns.mean - static_cast<double>(gap)) / p.cancel_gap_ns.stddev();
        } else {
            p.last_cancel_burst_zscore = 0.0;
        }
        p.cancel_gap_ns.update(static_cast<double>(gap));
    }
    p.last_cancel_ts_ns = ts_ns;
    p.has_last_cancel = true;

    if (const OrderMeta* meta = find_order_meta(order_id)) {
        p.lifetime_ns.update(static_cast<double>(ts_ns - meta->ts_added_ns));
        erase_order_meta(order_id);
    }
}

void FeatureEngine::on_execute(uint64_t participant_id, uint64_t order_id, uint64_t ts_ns) {
    ParticipantState& p = get_or_create_participant(participant_id);
    ++p.orders_executed;

    if (const OrderMeta* meta = find_order_meta(order_id)) {
        p.lifetime_ns.update(static_cast<double>(ts_ns - meta->ts_added_ns));
        erase_order_meta(order_id);
    }
}

ParticipantFeatureSnapshot FeatureEngine::snapshot(uint64_t participant_id) const {
    ParticipantFeatureSnapshot snap;
    const ParticipantState* p = find_participant(participant_id);
    if (p == nullptr) {
        return snap;
    }

    snap.order_to_trade_ratio =
        static_cast<double>(p->orders_placed) / static_cast<double>(std::max<uint64_t>(p->orders_executed, 1));
    snap.cancel_rate =
        static_cast<double>(p->orders_cancelled) / static_cast<double>(std::max<uint64_t>(p->orders_placed, 1));
    snap.mean_lifetime_ns = p->lifetime_ns.mean;
    snap.lifetime_stddev_ns = p->lifetime_ns.stddev();
    snap.layering_score_bid = p->layering_score_bid;
    snap.layering_score_ask = p->layering_score_ask;
    snap.cancel_burst_zscore = p->last_cancel_burst_zscore;
    snap.size_vs_baseline_ratio = p->last_size_vs_baseline_ratio;
    return snap;
}

} // namespace spoofwatch
