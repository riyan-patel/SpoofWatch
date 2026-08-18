#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "spoofwatch/lobster_message.hpp"

namespace spoofwatch {

// Parses a single LOBSTER message-file line (time,type,order_id,size,price,direction).
// Throws std::runtime_error on malformed input. Exposed for reuse by callers
// that need to walk a message file in lockstep with something else (e.g.
// the reference orderbook file for validation).
LobsterMessage parse_lobster_message_line(const std::string& line);

// Parses a comma-separated line of integers, as used by LOBSTER's
// orderbook reference file (AskPx1,AskSz1,BidPx1,BidSz1,...).
std::vector<int64_t> parse_int64_csv_line(const std::string& line);

// Streams a LOBSTER message CSV line-by-line and invokes `on_message` for
// each parsed row. Intentionally allocation-light: one line buffer reused
// across the whole file.
//
// Returns the number of messages parsed. Throws std::runtime_error if the
// file can't be opened or a line fails to parse.
size_t read_lobster_messages(
    const std::string& message_csv_path,
    const std::function<void(const LobsterMessage&)>& on_message);

} // namespace spoofwatch
