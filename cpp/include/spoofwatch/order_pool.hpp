#pragma once

#include <cstdint>
#include <vector>

#include "spoofwatch/order_record.hpp"

namespace spoofwatch {

// Fixed-capacity pool of OrderRecords plus an open-addressing hash map
// from order_id -> pool slot. All storage is allocated once at
// construction; allocate()/release()/find() never touch the heap.
//
// This replaces std::unordered_map in the hot path: no per-node
// allocation, no pointer chasing, cache-friendly linear probing.
class OrderPool {
public:
    explicit OrderPool(size_t capacity);

    // Inserts a new record for order_id and returns a pointer to it, or
    // nullptr if the pool is full or order_id already exists.
    OrderRecord* allocate(uint64_t order_id);

    // Returns the record for order_id, or nullptr if not present.
    OrderRecord* find(uint64_t order_id);
    const OrderRecord* find(uint64_t order_id) const;

    // Removes order_id from the pool, freeing its slot for reuse.
    // No-op if order_id is not present.
    void release(uint64_t order_id);

    size_t size() const { return live_count_; }
    size_t capacity() const { return records_.size(); }

private:
    enum class SlotState : uint8_t { Empty, Occupied, Tombstone };

    struct HashSlot {
        uint64_t order_id = 0;
        uint32_t pool_index = 0;
        SlotState state = SlotState::Empty;
    };

    size_t slot_for(uint64_t order_id) const;

    std::vector<OrderRecord> records_;
    std::vector<uint32_t> free_list_;
    size_t free_top_;
    size_t live_count_ = 0;

    std::vector<HashSlot> hash_slots_;
    size_t hash_mask_;
};

} // namespace spoofwatch
