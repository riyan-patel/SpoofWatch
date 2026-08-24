// Phase 6: per-stage latency + steady-state allocation benchmark for the
// assembled hot path (parse -> OrderBook -> FeatureEngine -> TreeModel).
// Reuses the same event loop as spoofwatch_pipeline, but times each stage
// with clock_gettime(CLOCK_MONOTONIC) into pre-allocated LatencyHistogram
// buckets, and counts heap allocations via the global operator new/delete
// override in alloc_counter.cpp — this binary is the only one those
// overrides are linked into, so no other tool or test is affected.
//
// The first `warmup_events` events are replayed normally but excluded
// from every measurement: this lets things like std::getline's reused
// line buffer finish growing to its steady-state capacity before the
// allocation counter is reset and the real measurement window begins.
// Note this benchmark's "parse" stage times CSV parsing, which stands in
// for the architecture diagram's Feed Handler stage — a real deployment
// would use a fixed-width binary wire format, not text CSV, so the parse
// numbers here are a conservative (slower) proxy, not a claim about how
// fast a production feed handler would be.
#include <charconv>
#include <cstdio>
#include <ctime>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "spoofwatch/alloc_counter.hpp"
#include "spoofwatch/feature_engine.hpp"
#include "spoofwatch/latency_histogram.hpp"
#include "spoofwatch/lobster_message.hpp"
#include "spoofwatch/order_book.hpp"
#include "spoofwatch/tree_model.hpp"

namespace {

using spoofwatch::LatencyHistogram;
using spoofwatch::LobsterEventType;

constexpr size_t kNumFeatures = 8;

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

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

void print_stage(const char* name, const LatencyHistogram& h) {
    std::printf(
        "  %-10s  n=%-8llu  min=%-6llu  mean=%-9.1f  p50=%-6llu  p99=%-7llu  p99.9=%-8llu  max=%llu  (ns)\n",
        name,
        static_cast<unsigned long long>(h.count()),
        static_cast<unsigned long long>(h.min_ns()),
        h.mean_ns(),
        static_cast<unsigned long long>(h.percentile(50)),
        static_cast<unsigned long long>(h.percentile(99)),
        static_cast<unsigned long long>(h.percentile(99.9)),
        static_cast<unsigned long long>(h.max_ns()));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <message_augmented.csv> <model.bin> [warmup_events=5000] "
            "[max_orders=2097152] [max_participants=65536]\n",
            argv[0]);
        std::fprintf(stderr,
            "Times each hot-path stage (parse/book/features/inference) and counts heap\n"
            "allocations after a warmup period, over a full replay of an injected file.\n");
        return 1;
    }

    std::string in_path = argv[1];
    std::string model_path = argv[2];
    size_t warmup_events = (argc >= 4) ? static_cast<size_t>(std::stoul(argv[3])) : 5000;
    size_t max_orders = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : (1u << 21);
    size_t max_participants = (argc >= 6) ? static_cast<size_t>(std::stoul(argv[5])) : (1u << 16);

    std::ifstream in(in_path);
    if (!in.is_open()) {
        std::fprintf(stderr, "Error: could not open input file: %s\n", in_path.c_str());
        return 1;
    }

    try {
        spoofwatch::TreeModel model(model_path);
        if (model.num_features() != kNumFeatures) {
            throw std::runtime_error("model feature count mismatch");
        }
        spoofwatch::OrderBook book(max_orders, /*max_price_levels=*/8192);
        spoofwatch::FeatureEngine engine(max_participants, max_orders);

        LatencyHistogram parse_hist, book_hist, feature_hist, inference_hist, total_new_hist;
        std::vector<spoofwatch::PriceLevel> asks, bids;
        double features[kNumFeatures];

        std::string line;
        bool first_line = true;
        size_t events_processed = 0;
        size_t orders_scored = 0;
        bool warmed_up = false;
        uint64_t measured_wall_start_ns = 0;

        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            if (first_line) {
                first_line = false; // header row from injector.py
                continue;
            }

            if (!warmed_up && events_processed == warmup_events) {
                spoofwatch::reset_alloc_count();
                measured_wall_start_ns = now_ns();
                warmed_up = true;
            }
            bool record = warmed_up;

            uint64_t t0 = now_ns();
            AugmentedRow row = parse_augmented_line(line);
            uint64_t t1 = now_ns();
            if (record) parse_hist.record(t1 - t0);

            auto event_type = static_cast<LobsterEventType>(row.type);
            spoofwatch::LobsterMessage msg{
                row.time_sec, event_type, row.order_id, row.size, row.price, row.direction};
            uint64_t ts_ns = static_cast<uint64_t>(row.time_sec * 1e9 + 0.5);
            spoofwatch::Side side =
                row.direction == 1 ? spoofwatch::Side::Bid : spoofwatch::Side::Ask;

            switch (event_type) {
                case LobsterEventType::NewLimitOrder: {
                    uint64_t t2 = now_ns();
                    book.top_n(1, asks, bids);
                    int64_t best_ask = asks.empty() ? row.price : asks[0].price;
                    int64_t best_bid = bids.empty() ? row.price : bids[0].price;
                    int64_t distance = (side == spoofwatch::Side::Bid)
                        ? (best_ask - row.price)
                        : (row.price - best_bid);
                    book.apply(msg);
                    uint64_t t3 = now_ns();
                    if (record) book_hist.record(t3 - t2);

                    engine.on_new_order(row.participant_id, row.order_id, side, row.size,
                                         distance, ts_ns);
                    spoofwatch::ParticipantFeatureSnapshot snap =
                        engine.snapshot(row.participant_id);
                    uint64_t t4 = now_ns();
                    if (record) feature_hist.record(t4 - t3);

                    snapshot_to_features(snap, features);
                    double proba = model.predict_proba(features, kNumFeatures);
                    (void)proba;
                    uint64_t t5 = now_ns();
                    if (record) inference_hist.record(t5 - t4);
                    if (record) total_new_hist.record(t5 - t0);
                    ++orders_scored;
                    break;
                }
                case LobsterEventType::PartialCancel:
                case LobsterEventType::Deletion: {
                    uint64_t t2 = now_ns();
                    book.apply(msg);
                    uint64_t t3 = now_ns();
                    if (record) book_hist.record(t3 - t2);
                    engine.on_cancel(row.participant_id, row.order_id, ts_ns);
                    uint64_t t4 = now_ns();
                    if (record) feature_hist.record(t4 - t3);
                    break;
                }
                case LobsterEventType::VisibleExecution:
                case LobsterEventType::HiddenExecution: {
                    uint64_t t2 = now_ns();
                    book.apply(msg);
                    uint64_t t3 = now_ns();
                    if (record) book_hist.record(t3 - t2);
                    engine.on_execute(row.participant_id, row.order_id, ts_ns);
                    uint64_t t4 = now_ns();
                    if (record) feature_hist.record(t4 - t3);
                    break;
                }
                case LobsterEventType::TradingHalt:
                default:
                    break;
            }
            ++events_processed;
        }

        uint64_t measured_wall_end_ns = now_ns();
        uint64_t measured_events = warmed_up ? events_processed - warmup_events : 0;
        uint64_t alloc_since_warmup = spoofwatch::alloc_count();

        std::printf("Replayed %zu events (%zu scored) from %s\n",
                    events_processed, orders_scored, in_path.c_str());
        std::printf("Warmup: %zu events (excluded from all stats below)\n", warmup_events);
        std::printf("Steady-state window: %llu events\n\n",
                     static_cast<unsigned long long>(measured_events));

        print_stage("parse", parse_hist);
        print_stage("book", book_hist);
        print_stage("features", feature_hist);
        print_stage("inference", inference_hist);
        print_stage("total(NEW)", total_new_hist);

        if (measured_events > 0 && measured_wall_end_ns > measured_wall_start_ns) {
            double wall_sec = static_cast<double>(measured_wall_end_ns - measured_wall_start_ns) / 1e9;
            std::printf("\nSteady-state throughput: %.0f events/sec over %.3fs\n",
                        static_cast<double>(measured_events) / wall_sec, wall_sec);
        }

        std::printf("\nHeap allocations since warmup ended: %llu — %s\n",
                    static_cast<unsigned long long>(alloc_since_warmup),
                    alloc_since_warmup == 0 ? "PASS (zero-allocation steady state)"
                                             : "FAIL (see note in pipeline.cpp/benchmark.cpp)");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
