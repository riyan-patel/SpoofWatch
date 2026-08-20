#include <gtest/gtest.h>

#include "spoofwatch/ring_buffer.hpp"

using spoofwatch::RingBuffer;

TEST(RingBuffer, EmptyInitially) {
    RingBuffer<int, 4> buf;
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.capacity(), 4u);
}

TEST(RingBuffer, PushBelowCapacityKeepsOrder) {
    RingBuffer<int, 4> buf;
    buf.push_back(1);
    buf.push_back(2);
    buf.push_back(3);
    ASSERT_EQ(buf.size(), 3u);
    EXPECT_EQ(buf[0], 1);
    EXPECT_EQ(buf[1], 2);
    EXPECT_EQ(buf[2], 3);
}

TEST(RingBuffer, OverflowEvictsOldest) {
    RingBuffer<int, 3> buf;
    buf.push_back(1);
    buf.push_back(2);
    buf.push_back(3);
    buf.push_back(4); // evicts 1
    buf.push_back(5); // evicts 2

    ASSERT_EQ(buf.size(), 3u);
    EXPECT_EQ(buf[0], 3);
    EXPECT_EQ(buf[1], 4);
    EXPECT_EQ(buf[2], 5);
}

TEST(RingBuffer, ClearResets) {
    RingBuffer<int, 3> buf;
    buf.push_back(1);
    buf.push_back(2);
    buf.clear();
    EXPECT_EQ(buf.size(), 0u);
    buf.push_back(9);
    EXPECT_EQ(buf[0], 9);
}
