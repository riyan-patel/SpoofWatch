#include "spoofwatch/price_level_book.hpp"

#include <algorithm>
#include <stdexcept>

namespace spoofwatch {

namespace {
constexpr int64_t kAskDummyPrice = 9999999999LL;
constexpr int64_t kBidDummyPrice = -9999999999LL;
} // namespace

PriceLevelBook::PriceLevelBook(size_t capacity, bool ascending)
    : capacity_(capacity), ascending_(ascending) {
    levels_.reserve(capacity); // one-time allocation; never reallocates after this
}

size_t PriceLevelBook::find_index(int64_t price, bool& found) const {
    auto cmp = [this](const PriceLevel& lvl, int64_t p) {
        return ascending_ ? lvl.price < p : lvl.price > p;
    };
    auto it = std::lower_bound(levels_.begin(), levels_.end(), price, cmp);
    found = (it != levels_.end() && it->price == price);
    return static_cast<size_t>(it - levels_.begin());
}

void PriceLevelBook::add_qty(int64_t price, uint64_t qty) {
    bool found = false;
    size_t idx = find_index(price, found);
    if (found) {
        levels_[idx].size += qty;
        return;
    }
    if (levels_.size() >= capacity_) {
        throw std::runtime_error("PriceLevelBook capacity exceeded");
    }
    levels_.insert(levels_.begin() + static_cast<long>(idx), PriceLevel{price, qty});
}

bool PriceLevelBook::remove_qty(int64_t price, uint64_t qty) {
    bool found = false;
    size_t idx = find_index(price, found);
    if (!found || levels_[idx].size < qty) {
        return false;
    }
    levels_[idx].size -= qty;
    if (levels_[idx].size == 0) {
        levels_.erase(levels_.begin() + static_cast<long>(idx));
    }
    return true;
}

void PriceLevelBook::top_n(size_t n, std::vector<PriceLevel>& out) const {
    out.clear();
    out.reserve(n);
    size_t have = std::min(n, levels_.size());
    for (size_t i = 0; i < have; ++i) {
        out.push_back(levels_[i]);
    }
    int64_t dummy_price = ascending_ ? kAskDummyPrice : kBidDummyPrice;
    for (size_t i = have; i < n; ++i) {
        out.push_back(PriceLevel{dummy_price, 0});
    }
}

} // namespace spoofwatch
