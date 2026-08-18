#include <gtest/gtest.h>

#include "spoofwatch/order_pool.hpp"

using spoofwatch::OrderPool;
using spoofwatch::OrderRecord;

TEST(OrderPool, AllocateFindRelease) {
    OrderPool pool(16);

    OrderRecord* rec = pool.allocate(42);
    ASSERT_NE(rec, nullptr);
    EXPECT_EQ(rec->order_id, 42u);
    EXPECT_EQ(pool.size(), 1u);

    OrderRecord* found = pool.find(42);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found, rec);

    pool.release(42);
    EXPECT_EQ(pool.size(), 0u);
    EXPECT_EQ(pool.find(42), nullptr);
}

TEST(OrderPool, DuplicateAllocateFails) {
    OrderPool pool(16);
    ASSERT_NE(pool.allocate(1), nullptr);
    EXPECT_EQ(pool.allocate(1), nullptr);
}

TEST(OrderPool, FindMissingReturnsNull) {
    OrderPool pool(16);
    EXPECT_EQ(pool.find(999), nullptr);
}

TEST(OrderPool, ReleaseThenReallocateReusesSlot) {
    OrderPool pool(4);
    pool.allocate(1);
    pool.allocate(2);
    pool.release(1);

    OrderRecord* rec = pool.allocate(3);
    ASSERT_NE(rec, nullptr);
    EXPECT_EQ(pool.size(), 2u);
    EXPECT_EQ(pool.find(1), nullptr);
    EXPECT_NE(pool.find(2), nullptr);
    EXPECT_NE(pool.find(3), nullptr);
}

TEST(OrderPool, ExhaustionReturnsNull) {
    OrderPool pool(2);
    ASSERT_NE(pool.allocate(1), nullptr);
    ASSERT_NE(pool.allocate(2), nullptr);
    EXPECT_EQ(pool.allocate(3), nullptr);
}

TEST(OrderPool, ManyOrdersRoundTrip) {
    constexpr size_t kN = 1000;
    OrderPool pool(kN);
    for (uint64_t id = 1; id <= kN; ++id) {
        ASSERT_NE(pool.allocate(id), nullptr) << "id=" << id;
    }
    for (uint64_t id = 1; id <= kN; ++id) {
        OrderRecord* rec = pool.find(id);
        ASSERT_NE(rec, nullptr) << "id=" << id;
        EXPECT_EQ(rec->order_id, id);
    }
}
