#pragma once

#include <array>
#include <cstddef>

namespace spoofwatch {

// Fixed-capacity circular buffer. push_back() overwrites the oldest entry
// once full — no heap allocation, no shifting. operator[](0) is the
// oldest surviving entry, operator[](size()-1) is the newest.
template <typename T, size_t Capacity>
class RingBuffer {
public:
    void push_back(const T& value) {
        data_[next_] = value;
        next_ = (next_ + 1) % Capacity;
        if (count_ < Capacity) {
            ++count_;
        }
    }

    void clear() {
        next_ = 0;
        count_ = 0;
    }

    size_t size() const { return count_; }
    constexpr size_t capacity() const { return Capacity; }
    bool empty() const { return count_ == 0; }

    const T& operator[](size_t logical_index) const {
        size_t start = (count_ == Capacity) ? next_ : 0;
        return data_[(start + logical_index) % Capacity];
    }

private:
    std::array<T, Capacity> data_{};
    size_t next_ = 0;
    size_t count_ = 0;
};

} // namespace spoofwatch
