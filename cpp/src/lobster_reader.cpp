#include "spoofwatch/lobster_reader.hpp"

#include <charconv>
#include <fstream>
#include <stdexcept>

namespace spoofwatch {

namespace {

// Parses the next comma-separated field starting at `pos` in `line`,
// advancing `pos` past the delimiter. Returns the field as a string_view.
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
    // std::from_chars for double needs a null-terminated-safe buffer on
    // some stdlibs; string_view -> std::string is cheap here since this
    // is Phase 0 parsing, not the hot path.
    try {
        size_t consumed = 0;
        double value = std::stod(std::string(field), &consumed);
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Failed to parse time field '" + std::string(field) +
                                  "' in line: " + line);
    }
}

} // namespace

LobsterMessage parse_lobster_message_line(const std::string& line) {
    LobsterMessage msg{};
    size_t pos = 0;

    msg.time_sec = parse_time(next_field(line, pos), line);
    msg.type = static_cast<LobsterEventType>(parse_number<int32_t>(next_field(line, pos), line));
    msg.order_id = parse_number<uint64_t>(next_field(line, pos), line);
    msg.size = parse_number<uint32_t>(next_field(line, pos), line);
    msg.price = parse_number<int64_t>(next_field(line, pos), line);
    msg.direction = parse_number<int32_t>(next_field(line, pos), line);

    return msg;
}

std::vector<int64_t> parse_int64_csv_line(const std::string& line) {
    std::vector<int64_t> values;
    size_t pos = 0;
    while (pos < line.size()) {
        values.push_back(parse_number<int64_t>(next_field(line, pos), line));
    }
    return values;
}

size_t read_lobster_messages(
    const std::string& message_csv_path,
    const std::function<void(const LobsterMessage&)>& on_message) {
    std::ifstream file(message_csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open LOBSTER message file: " + message_csv_path);
    }

    std::string line;
    size_t count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        on_message(parse_lobster_message_line(line));
        ++count;
    }

    return count;
}

} // namespace spoofwatch
