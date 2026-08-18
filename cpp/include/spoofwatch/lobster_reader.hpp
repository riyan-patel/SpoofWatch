#pragma once

#include <functional>
#include <string>

#include "spoofwatch/lobster_message.hpp"

namespace spoofwatch {

// Streams a LOBSTER message CSV line-by-line and invokes `on_message` for
// each parsed row. Intentionally allocation-light: one line buffer reused
// across the whole file. This is Phase 0 scaffolding (parse + print) — it
// does not build book state, that lands in Phase 1.
//
// Returns the number of messages parsed. Throws std::runtime_error if the
// file can't be opened or a line fails to parse.
size_t read_lobster_messages(
    const std::string& message_csv_path,
    const std::function<void(const LobsterMessage&)>& on_message);

} // namespace spoofwatch
