// Replays a participant-labeled, synthetic-pattern-injected message file
// (as produced by python/injection/injector.py) through OrderBook and
// FeatureEngine, emitting one feature-vector row per order lifecycle
// event. This is the bridge between Phase 3 (ground-truth injection) and
// Phase 4 (model training): the same FeatureEngine that will eventually
// sit in the hot inference path is the one generating training data, so
// there's no risk of the Python training pipeline and the C++ engine
// disagreeing on what a feature means.
//
// Input schema (header + 7 comma-separated columns):
//   time,type,order_id,size,price,direction,participant_id
// (LOBSTER's 6-column schema plus the synthetic participant_id column
// added by the injection pipeline.)
//
// Labeling against ground_truth.csv happens downstream in Python — this
// tool only computes features, it doesn't know which orders are
// manipulative.
#include <charconv>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "spoofwatch/feature_engine.hpp"
#include "spoofwatch/lobster_message.hpp"
#include "spoofwatch/order_book.hpp"

namespace {

using spoofwatch::LobsterEventType;

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

const char* event_label(int32_t type) {
    switch (static_cast<LobsterEventType>(type)) {
        case LobsterEventType::NewLimitOrder: return "NEW";
        case LobsterEventType::PartialCancel: return "PARTIAL_CANCEL";
        case LobsterEventType::Deletion: return "DELETE";
        case LobsterEventType::VisibleExecution: return "EXEC_VISIBLE";
        case LobsterEventType::HiddenExecution: return "EXEC_HIDDEN";
        case LobsterEventType::TradingHalt: return "TRADING_HALT";
        default: return "UNKNOWN";
    }
}

void write_snapshot_row(
    std::ofstream& out, const AugmentedRow& row, const char* event,
    spoofwatch::Side side, const spoofwatch::ParticipantFeatureSnapshot& snap) {
    out << row.time_sec << ',' << event << ',' << row.order_id << ','
        << row.participant_id << ',' << (side == spoofwatch::Side::Bid ? "BID" : "ASK") << ','
        << snap.order_to_trade_ratio << ',' << snap.cancel_rate << ','
        << snap.mean_lifetime_ns << ',' << snap.lifetime_stddev_ns << ','
        << snap.layering_score_bid << ',' << snap.layering_score_ask << ','
        << snap.cancel_burst_zscore << ',' << snap.size_vs_baseline_ratio << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <message_augmented.csv> <features_out.csv> [max_orders=2097152] [max_participants=65536]\n",
            argv[0]);
        std::fprintf(stderr,
            "Replays an injection-pipeline output file through OrderBook + FeatureEngine\n"
            "and writes one feature-vector row per order lifecycle event.\n");
        return 1;
    }

    std::string in_path = argv[1];
    std::string out_path = argv[2];
    size_t max_orders = (argc >= 4) ? static_cast<size_t>(std::stoul(argv[3])) : (1u << 21);
    size_t max_participants = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : (1u << 16);

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

    spoofwatch::OrderBook book(max_orders, /*max_price_levels=*/8192);
    spoofwatch::FeatureEngine engine(max_participants, max_orders);

    out << "time,event,order_id,participant_id,side,order_to_trade_ratio,cancel_rate,"
           "mean_lifetime_ns,lifetime_stddev_ns,layering_score_bid,layering_score_ask,"
           "cancel_burst_zscore,size_vs_baseline_ratio\n";

    std::string line;
    bool first = true;
    size_t events_processed = 0;
    size_t rows_written = 0;
    std::vector<spoofwatch::PriceLevel> asks, bids;

    try {
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            if (first) {
                // Skip the header row written by injector.py.
                first = false;
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
                    write_snapshot_row(out, row, event_label(row.type), side,
                                        engine.snapshot(row.participant_id));
                    ++rows_written;
                    break;
                }
                case LobsterEventType::PartialCancel:
                case LobsterEventType::Deletion: {
                    engine.on_cancel(row.participant_id, row.order_id, ts_ns);
                    book.apply(msg);
                    write_snapshot_row(out, row, event_label(row.type), side,
                                        engine.snapshot(row.participant_id));
                    ++rows_written;
                    break;
                }
                case LobsterEventType::VisibleExecution:
                case LobsterEventType::HiddenExecution: {
                    engine.on_execute(row.participant_id, row.order_id, ts_ns);
                    book.apply(msg);
                    write_snapshot_row(out, row, event_label(row.type), side,
                                        engine.snapshot(row.participant_id));
                    ++rows_written;
                    break;
                }
                case LobsterEventType::TradingHalt:
                default:
                    break;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    std::printf("Processed %zu events, wrote %zu feature rows to %s\n",
                events_processed, rows_written, out_path.c_str());
    return 0;
}
