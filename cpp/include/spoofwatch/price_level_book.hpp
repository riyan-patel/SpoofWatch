#pragma once

#include <cstdint>
#include <vector>

namespace spoofwatch {

struct PriceLevel {
    int64_t price;   // dollars * 10000, matching LOBSTER's message format
    uint64_t size;   // total resting shares at this price
};

// One side (bid or ask) of the book: a fixed-capacity array of price
// levels kept sorted by `better_than`. Deliberately not a std::map — at
// the depths a single symbol's book reaches, a flat sorted array with
// binary search + memmove beats a red-black tree on cache behavior, and
// it never allocates past construction.
class PriceLevelBook {
public:
    // better_than(a, b) should return true if price `a` is closer to the
    // touch than price `b` (e.g. descending for bids, ascending for asks).
    PriceLevelBook(size_t capacity, bool ascending);

    void add_qty(int64_t price, uint64_t qty);
    // Returns false if the price level doesn't exist or has insufficient qty.
    bool remove_qty(int64_t price, uint64_t qty);

    size_t level_count() const { return levels_.size(); }
    const PriceLevel& level_at(size_t i) const { return levels_[i]; }

    // Copies up to `n` best levels into `out`, padding with LOBSTER's
    // dummy sentinel (+/-9999999999 price, 0 size) if fewer levels exist.
    void top_n(size_t n, std::vector<PriceLevel>& out) const;

private:
    size_t find_index(int64_t price, bool& found) const;

    std::vector<PriceLevel> levels_;
    size_t capacity_;
    bool ascending_;
};

} // namespace spoofwatch
