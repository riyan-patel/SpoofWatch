// Phase 6 groundwork: the full hot path assembled into one process —
// parse -> OrderBook -> FeatureEngine -> TreeModel inference — matching
// the architecture diagram's stages [1]-[4] in README.md. Previously
// these were exercised as three separate tools chained through files
// (spoofwatch_features, then Python training, then spoofwatch_infer_validate);
// this is the actual single-pass loop that per-stage latency
// instrumentation will be added to next.
//
// Input schema matches spoofwatch_features (header + 7 columns):
//   time,type,order_id,size,price,direction,participant_id
// Only NEW events are scored — that's the only point in an order's
// lifecycle a real-time system would act on, matching how the model was
// trained (see python/training/dataset.py).
#include <charconv>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "spoofwatch/feature_engine.hpp"
#include "spoofwatch/lobster_message.hpp"
#include "spoofwatch/order_book.hpp"
#include "spoofwatch/tree_model.hpp"

namespace {

using spoofwatch::LobsterEventType;

constexpr size_t kNumFeatures = 8; // must match python/training/dataset.py FEATURE_COLUMNS

struct AugmentedRow {
    double time_sec;
    int32_t type;
    uint64_t order_id;
    uint32_t size;
    int64_t price;
    int32_t direction;
    uint64_t participant_id;
};

std::string_view next_field(const std::string& line, size_t& pos) {
    size_t start = pos;
    size_t comma = line.find(',', start);
    if (comma == std::string::npos) {
        pos = line.size();
        return std::string_view(line).substr(start);
    }
    pos = comma + 1;
    return std::string_view(line).substr(start, comma - start);
}

template <typename T>
T parse_number(std::string_view field, const std::string& line) {
    T value{};
    auto result = std::from_chars(field.data(), field.data() + field.size(), value);
    if (result.ec != std::errc()) {
        throw std::runtime_error("Failed to parse field '" + std::string(field) +
                                  "' in line: " + line);
    }
    return value;
}

double parse_time(std::string_view field, const std::string& line) {
    try {
        return std::stod(std::string(field));
    } catch (const std::exception&) {
        throw std::runtime_error("Failed to parse time field '" + std::string(field) +
                                  "' in line: " + line);
    }
}

AugmentedRow parse_augmented_line(const std::string& line) {
    AugmentedRow row{};
    size_t pos = 0;
    row.time_sec = parse_time(next_field(line, pos), line);
    row.type = parse_number<int32_t>(next_field(line, pos), line);
    row.order_id = parse_number<uint64_t>(next_field(line, pos), line);
    row.size = parse_number<uint32_t>(next_field(line, pos), line);
    row.price = parse_number<int64_t>(next_field(line, pos), line);
    row.direction = parse_number<int32_t>(next_field(line, pos), line);
    row.participant_id = parse_number<uint64_t>(next_field(line, pos), line);
    return row;
}

// Field order here must match python/training/dataset.py's FEATURE_COLUMNS
// exactly — it happens to already match ParticipantFeatureSnapshot's
// declaration order, which is what keeps this a straight field-by-field
// copy rather than something that needs a name-indexed lookup.
void snapshot_to_features(const spoofwatch::ParticipantFeatureSnapshot& snap,
                           double (&out)[kNumFeatures]) {
    out[0] = snap.order_to_trade_ratio;
    out[1] = snap.cancel_rate;
    out[2] = snap.mean_lifetime_ns;
    out[3] = snap.lifetime_stddev_ns;
    out[4] = static_cast<double>(snap.layering_score_bid);
    out[5] = static_cast<double>(snap.layering_score_ask);
    out[6] = snap.cancel_burst_zscore;
    out[7] = snap.size_vs_baseline_ratio;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <message_augmented.csv> <model.bin> <scored_out.csv> "
            "[max_orders=2097152] [max_participants=65536]\n",
            argv[0]);
        std::fprintf(stderr,
            "Replays an injection-pipeline output file through the full hot path —\n"
            "OrderBook -> FeatureEngine -> TreeModel — scoring every NEW order.\n");
        return 1;
    }

    std::string in_path = argv[1];
    std::string model_path = argv[2];
    std::string out_path = argv[3];
    size_t max_orders = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : (1u << 21);
    size_t max_participants = (argc >= 6) ? static_cast<size_t>(std::stoul(argv[5])) : (1u << 16);

    std::ifstream in(in_path);
    if (!in.is_open()) {
        std::fprintf(stderr, "Error: could not open input file: %s\n", in_path.c_str());
        return 1;
    }
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Error: could not open output file: %s\n", out_path.c_str());
        return 1;
    }

    try {
        spoofwatch::TreeModel model(model_path);
        if (model.num_features() != kNumFeatures) {
            throw std::runtime_error(
                "model expects " + std::to_string(model.num_features()) +
                " features but this pipeline produces " + std::to_string(kNumFeatures));
        }

        spoofwatch::OrderBook book(max_orders, /*max_price_levels=*/8192);
        spoofwatch::FeatureEngine engine(max_participants, max_orders);

        out << "time,order_id,participant_id,order_to_trade_ratio,cancel_rate,"
               "mean_lifetime_ns,lifetime_stddev_ns,layering_score_bid,layering_score_ask,"
               "cancel_burst_zscore,size_vs_baseline_ratio,proba\n";

        std::string line;
        bool first = true;
        size_t events_processed = 0;
        size_t orders_scored = 0;
        double proba_sum = 0.0;
        std::vector<spoofwatch::PriceLevel> asks, bids;
        double features[kNumFeatures];

        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            if (first) {
                first = false; // header row from injector.py
                continue;
            }
            AugmentedRow row = parse_augmented_line(line);
            ++events_processed;

            auto event_type = static_cast<LobsterEventType>(row.type);
            spoofwatch::LobsterMessage msg{
                row.time_sec, event_type, row.order_id, row.size, row.price, row.direction};
            uint64_t ts_ns = static_cast<uint64_t>(row.time_sec * 1e9 + 0.5);
            spoofwatch::Side side =
                row.direction == 1 ? spoofwatch::Side::Bid : spoofwatch::Side::Ask;

            switch (event_type) {
                case LobsterEventType::NewLimitOrder: {
                    book.top_n(1, asks, bids);
                    int64_t best_ask = asks.empty() ? row.price : asks[0].price;
                    int64_t best_bid = bids.empty() ? row.price : bids[0].price;
                    int64_t distance = (side == spoofwatch::Side::Bid)
                        ? (best_ask - row.price)
                        : (row.price - best_bid);
                    engine.on_new_order(row.participant_id, row.order_id, side, row.size,
                                         distance, ts_ns);
                    book.apply(msg);

                    spoofwatch::ParticipantFeatureSnapshot snap =
                        engine.snapshot(row.participant_id);
                    snapshot_to_features(snap, features);
                    double proba = model.predict_proba(features, kNumFeatures);

                    out << row.time_sec << ',' << row.order_id << ',' << row.participant_id;
                    for (double f : features) {
                        out << ',' << f;
                    }
                    out << ',' << proba << '\n';
                    proba_sum += proba;
                    ++orders_scored;
                    break;
                }
                case LobsterEventType::PartialCancel:
                case LobsterEventType::Deletion:
                    engine.on_cancel(row.participant_id, row.order_id, ts_ns);
                    book.apply(msg);
                    break;
                case LobsterEventType::VisibleExecution:
                case LobsterEventType::HiddenExecution:
                    engine.on_execute(row.participant_id, row.order_id, ts_ns);
                    book.apply(msg);
                    break;
                case LobsterEventType::TradingHalt:
                default:
                    break;
            }
        }

        std::printf(
            "Processed %zu events, scored %zu orders (mean proba %.6f), wrote %s\n",
            events_processed, orders_scored, orders_scored ? proba_sum / orders_scored : 0.0,
            out_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
