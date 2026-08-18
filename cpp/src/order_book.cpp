#include "spoofwatch/order_book.hpp"

namespace spoofwatch {

namespace {
uint64_t to_ns(double time_sec) {
    return static_cast<uint64_t>(time_sec * 1e9);
}
} // namespace

OrderBook::OrderBook(size_t max_orders, size_t max_price_levels)
    : pool_(max_orders),
      bids_(max_price_levels, /*ascending=*/false),
      asks_(max_price_levels, /*ascending=*/true) {}

void OrderBook::apply(const LobsterMessage& msg) {
    switch (msg.type) {
        case LobsterEventType::NewLimitOrder: {
            OrderRecord* record = pool_.allocate(msg.order_id);
            if (record == nullptr) {
                return; // pool exhausted or duplicate id — nothing more we can do
            }
            record->price = static_cast<double>(msg.price) / 10000.0;
            record->size = msg.size;
            record->side = (msg.direction == 1) ? Side::Bid : Side::Ask;
            record->status = OrderStatus::Active;
            record->ts_added_ns = to_ns(msg.time_sec);
            record->ts_last_event_ns = record->ts_added_ns;

            auto& side_book = (record->side == Side::Bid) ? bids_ : asks_;
            side_book.add_qty(msg.price, msg.size);
            break;
        }
        case LobsterEventType::PartialCancel: {
            OrderRecord* record = pool_.find(msg.order_id);
            if (record == nullptr) {
                return;
            }
            auto& side_book = (record->side == Side::Bid) ? bids_ : asks_;
            side_book.remove_qty(msg.price, msg.size);
            record->size = (msg.size < record->size) ? record->size - msg.size : 0;
            record->status = OrderStatus::Partial;
            record->ts_last_event_ns = to_ns(msg.time_sec);
            break;
        }
        case LobsterEventType::Deletion: {
            OrderRecord* record = pool_.find(msg.order_id);
            if (record == nullptr) {
                return;
            }
            auto& side_book = (record->side == Side::Bid) ? bids_ : asks_;
            side_book.remove_qty(msg.price, record->size);
            pool_.release(msg.order_id);
            break;
        }
        case LobsterEventType::VisibleExecution: {
            OrderRecord* record = pool_.find(msg.order_id);
            if (record == nullptr) {
                return;
            }
            auto& side_book = (record->side == Side::Bid) ? bids_ : asks_;
            side_book.remove_qty(msg.price, msg.size);
            record->size = (msg.size < record->size) ? record->size - msg.size : 0;
            record->ts_last_event_ns = to_ns(msg.time_sec);
            if (record->size == 0) {
                record->status = OrderStatus::Executed;
                pool_.release(msg.order_id);
            } else {
                record->status = OrderStatus::Partial;
            }
            break;
        }
        case LobsterEventType::HiddenExecution:
            // Hidden orders were never added to the visible book (no
            // NewLimitOrder preceded them), so there's nothing to update.
            break;
        case LobsterEventType::TradingHalt:
            // Per LOBSTER's spec, the reference orderbook row simply
            // repeats the prior state for these messages.
            break;
    }
}

void OrderBook::top_n(size_t n, std::vector<PriceLevel>& asks, std::vector<PriceLevel>& bids) const {
    asks_.top_n(n, asks);
    bids_.top_n(n, bids);
}

} // namespace spoofwatch
