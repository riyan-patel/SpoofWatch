#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "spoofwatch/lobster_reader.hpp"
#include "spoofwatch/order_book.hpp"

namespace {

struct RowStats {
    bool full_row_match = true;
    bool level1_match = true;
    size_t fields_matched = 0;
    size_t fields_total = 0;
};

// Compares our reconstructed top-n levels against one row of LOBSTER's
// orderbook reference file: AskPx1,AskSz1,BidPx1,BidSz1,AskPx2,... .
//
// NOTE: LOBSTER's message file only records events for orders that affect
// the top-N price levels at the time of the event. An order can start
// inside the top N, get buried deeper as later orders arrive, and then be
// cancelled/executed with no corresponding message in a depth-limited
// file. The opening-auction event that seeds the very first row is also
// excluded from public "clean" samples. Both effects mean full-day exact
// reconstruction from a depth-limited message file is not achievable —
// this comparator reports match statistics as a diagnostic, not a
// pass/fail gate.
RowStats compare_row(
    const std::vector<int64_t>& expected,
    const std::vector<spoofwatch::PriceLevel>& asks,
    const std::vector<spoofwatch::PriceLevel>& bids) {
    RowStats stats;
    for (size_t level = 0; level < asks.size(); ++level) {
        size_t base = level * 4;
        if (base + 3 >= expected.size()) {
            stats.full_row_match = false;
            break;
        }
        bool ap_ok = expected[base + 0] == asks[level].price;
        bool az_ok = static_cast<uint64_t>(expected[base + 1]) == asks[level].size;
        bool bp_ok = expected[base + 2] == bids[level].price;
        bool bz_ok = static_cast<uint64_t>(expected[base + 3]) == bids[level].size;

        stats.fields_total += 4;
        stats.fields_matched += ap_ok + az_ok + bp_ok + bz_ok;

        if (!(ap_ok && az_ok && bp_ok && bz_ok)) {
            stats.full_row_match = false;
        }
        if (level == 0) {
            stats.level1_match = ap_ok && az_ok && bp_ok && bz_ok;
        }
    }
    return stats;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <message.csv> <orderbook.csv> [num_levels=10]\n", argv[0]);
        std::fprintf(stderr, "Replays a LOBSTER message file through OrderBook and diffs the\n");
        std::fprintf(stderr, "result against the reference orderbook file, row by row.\n");
        return 1;
    }

    std::string message_path = argv[1];
    std::string orderbook_path = argv[2];
    size_t num_levels = (argc >= 4) ? static_cast<size_t>(std::stoul(argv[3])) : 10;

    std::ifstream ob_file(orderbook_path);
    if (!ob_file.is_open()) {
        std::fprintf(stderr, "Error: could not open orderbook file: %s\n", orderbook_path.c_str());
        return 1;
    }

    spoofwatch::OrderBook book(/*max_orders=*/1u << 21, /*max_price_levels=*/8192);

    size_t row_index = 0;
    size_t full_row_matches = 0;
    size_t level1_matches = 0;
    size_t fields_matched_total = 0;
    size_t fields_total_total = 0;
    std::vector<spoofwatch::PriceLevel> asks, bids;
    std::string ob_line;

    try {
        size_t total = spoofwatch::read_lobster_messages(message_path, [&](const spoofwatch::LobsterMessage& msg) {
            book.apply(msg);

            if (!std::getline(ob_file, ob_line)) {
                throw std::runtime_error("orderbook file has fewer rows than message file");
            }
            auto expected = spoofwatch::parse_int64_csv_line(ob_line);

            book.top_n(num_levels, asks, bids);
            RowStats stats = compare_row(expected, asks, bids);
            full_row_matches += stats.full_row_match;
            level1_matches += stats.level1_match;
            fields_matched_total += stats.fields_matched;
            fields_total_total += stats.fields_total;
            ++row_index;
        });

        std::printf("Replayed %zu events, compared %zu rows against LOBSTER's reference orderbook.\n", total, row_index);
        std::printf("  full-row exact match:  %zu/%zu (%.2f%%)\n",
                     full_row_matches, row_index, 100.0 * full_row_matches / row_index);
        std::printf("  best bid/ask match:    %zu/%zu (%.2f%%)\n",
                     level1_matches, row_index, 100.0 * level1_matches / row_index);
        std::printf("  per-field match:       %zu/%zu (%.2f%%)\n",
                     fields_matched_total, fields_total_total,
                     100.0 * fields_matched_total / fields_total_total);
        std::printf(
            "\nNote: LOBSTER's message file only records events for orders that affect\n"
            "the top-%zu levels at the time of the event, and the opening-auction event\n"
            "that seeds the first row is excluded from public samples. Exact full-day\n"
            "reconstruction from a depth-limited file is not achievable by design; these\n"
            "numbers are a diagnostic, not a pass/fail gate.\n",
            num_levels);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
