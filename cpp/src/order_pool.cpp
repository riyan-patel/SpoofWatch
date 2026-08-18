#include "spoofwatch/order_pool.hpp"

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

// Fibonacci-style multiplicative hash — cheap, good enough for order IDs
// which LOBSTER assigns in monotonically increasing order flow.
uint64_t hash_order_id(uint64_t order_id) {
    uint64_t h = order_id;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

} // namespace

OrderPool::OrderPool(size_t capacity) : records_(capacity), free_top_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("OrderPool capacity must be > 0");
    }
    free_list_.resize(capacity);
    for (size_t i = 0; i < capacity; ++i) {
        // Fill so index 0 pops first — order doesn't matter functionally.
        free_list_[i] = static_cast<uint32_t>(capacity - 1 - i);
    }

    // Load factor <= 0.5 keeps linear-probing chains short.
    size_t hash_capacity = next_power_of_two(capacity * 2);
    hash_slots_.resize(hash_capacity);
    hash_mask_ = hash_capacity - 1;
}

size_t OrderPool::slot_for(uint64_t order_id) const {
    return hash_order_id(order_id) & hash_mask_;
}

OrderRecord* OrderPool::allocate(uint64_t order_id) {
    if (free_top_ == 0) {
        return nullptr; // pool exhausted
    }

    size_t idx = slot_for(order_id);
    size_t first_tombstone = hash_slots_.size(); // sentinel: none found
    for (size_t probes = 0; probes < hash_slots_.size(); ++probes) {
        HashSlot& slot = hash_slots_[idx];
        if (slot.state == SlotState::Occupied) {
            if (slot.order_id == order_id) {
                return nullptr; // duplicate order_id
            }
        } else {
            if (slot.state == SlotState::Tombstone && first_tombstone == hash_slots_.size()) {
                first_tombstone = idx;
            }
            if (slot.state == SlotState::Empty) {
                size_t target = (first_tombstone != hash_slots_.size()) ? first_tombstone : idx;
                uint32_t pool_index = free_list_[--free_top_];

                hash_slots_[target] = HashSlot{order_id, pool_index, SlotState::Occupied};
                records_[pool_index] = OrderRecord{};
                records_[pool_index].order_id = order_id;
                ++live_count_;
                return &records_[pool_index];
            }
        }
        idx = (idx + 1) & hash_mask_;
    }
    return nullptr; // hash table full (shouldn't happen given 2x sizing)
}

OrderRecord* OrderPool::find(uint64_t order_id) {
    size_t idx = slot_for(order_id);
    for (size_t probes = 0; probes < hash_slots_.size(); ++probes) {
        HashSlot& slot = hash_slots_[idx];
        if (slot.state == SlotState::Empty) {
            return nullptr;
        }
        if (slot.state == SlotState::Occupied && slot.order_id == order_id) {
            return &records_[slot.pool_index];
        }
        idx = (idx + 1) & hash_mask_;
    }
    return nullptr;
}

const OrderRecord* OrderPool::find(uint64_t order_id) const {
    return const_cast<OrderPool*>(this)->find(order_id);
}

void OrderPool::release(uint64_t order_id) {
    size_t idx = slot_for(order_id);
    for (size_t probes = 0; probes < hash_slots_.size(); ++probes) {
        HashSlot& slot = hash_slots_[idx];
        if (slot.state == SlotState::Empty) {
            return; // not present
        }
        if (slot.state == SlotState::Occupied && slot.order_id == order_id) {
            free_list_[free_top_++] = slot.pool_index;
            slot.state = SlotState::Tombstone;
            --live_count_;
            return;
        }
        idx = (idx + 1) & hash_mask_;
    }
}

} // namespace spoofwatch
