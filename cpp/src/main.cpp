#include <cstdio>
#include <exception>
#include <string>

#include "spoofwatch/lobster_reader.hpp"

namespace {

const char* event_type_name(spoofwatch::LobsterEventType type) {
    using spoofwatch::LobsterEventType;
    switch (type) {
        case LobsterEventType::NewLimitOrder: return "NEW";
        case LobsterEventType::PartialCancel: return "PARTIAL_CANCEL";
        case LobsterEventType::Deletion: return "DELETE";
        case LobsterEventType::VisibleExecution: return "EXEC_VISIBLE";
        case LobsterEventType::HiddenExecution: return "EXEC_HIDDEN";
        case LobsterEventType::TradingHalt: return "TRADING_HALT";
        default: return "UNKNOWN";
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <lobster_message.csv> [max_print]\n", argv[0]);
        std::fprintf(stderr, "Phase 0 scaffold: parses a LOBSTER message file and prints structured events.\n");
        return 1;
    }

    std::string path = argv[1];
    size_t max_print = (argc >= 3) ? static_cast<size_t>(std::stoul(argv[2])) : 20;

    try {
        size_t printed = 0;
        size_t total = spoofwatch::read_lobster_messages(path, [&](const spoofwatch::LobsterMessage& msg) {
            if (printed < max_print) {
                std::printf(
                    "t=%.9f  %-14s order_id=%-10llu size=%-6u price=%-9lld dir=%+d\n",
                    msg.time_sec,
                    event_type_name(msg.type),
                    static_cast<unsigned long long>(msg.order_id),
                    msg.size,
                    static_cast<long long>(msg.price),
                    msg.direction);
                ++printed;
            }
        });

        std::printf("\nParsed %zu total events from %s (printed first %zu).\n", total, path.c_str(), printed);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
