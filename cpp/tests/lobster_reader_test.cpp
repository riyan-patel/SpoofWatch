#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <vector>
#include <unistd.h>

#include "spoofwatch/lobster_reader.hpp"

namespace {

std::string write_temp_csv(const std::string& contents) {
    char path_template[] = "/tmp/spoofwatch_test_XXXXXX";
    int fd = mkstemp(path_template);
    close(fd);
    std::string path = path_template;
    std::ofstream file(path);
    file << contents;
    file.close();
    return path;
}

} // namespace

TEST(LobsterReader, ParsesAllColumnsCorrectly) {
    // One NEW and one DELETE row, matching the real LOBSTER column order:
    // time,type,order_id,size,price,direction
    std::string path = write_temp_csv(
        "34200.004241176,1,16113575,18,5853300,1\n"
        "34200.025551909,3,16120456,18,5859100,-1\n");

    std::vector<spoofwatch::LobsterMessage> messages;
    size_t count = spoofwatch::read_lobster_messages(path, [&](const spoofwatch::LobsterMessage& msg) {
        messages.push_back(msg);
    });
    std::remove(path.c_str());

    ASSERT_EQ(count, 2u);
    ASSERT_EQ(messages.size(), 2u);

    EXPECT_DOUBLE_EQ(messages[0].time_sec, 34200.004241176);
    EXPECT_EQ(messages[0].type, spoofwatch::LobsterEventType::NewLimitOrder);
    EXPECT_EQ(messages[0].order_id, 16113575u);
    EXPECT_EQ(messages[0].size, 18u);
    EXPECT_EQ(messages[0].price, 5853300);
    EXPECT_EQ(messages[0].direction, 1);

    EXPECT_EQ(messages[1].type, spoofwatch::LobsterEventType::Deletion);
    EXPECT_EQ(messages[1].order_id, 16120456u);
    EXPECT_EQ(messages[1].direction, -1);
}

TEST(LobsterReader, SkipsBlankLines) {
    std::string path = write_temp_csv(
        "34200.004241176,1,16113575,18,5853300,1\n"
        "\n"
        "34200.025551909,3,16120456,18,5859100,-1\n");

    size_t count = spoofwatch::read_lobster_messages(path, [](const spoofwatch::LobsterMessage&) {});
    std::remove(path.c_str());

    EXPECT_EQ(count, 2u);
}

TEST(LobsterReader, ThrowsOnMissingFile) {
    EXPECT_THROW(
        spoofwatch::read_lobster_messages("/nonexistent/path/does_not_exist.csv",
                                           [](const spoofwatch::LobsterMessage&) {}),
        std::runtime_error);
}

TEST(LobsterReader, ThrowsOnMalformedLine) {
    std::string path = write_temp_csv("not,a,valid,lobster,row\n");
    EXPECT_THROW(
        spoofwatch::read_lobster_messages(path, [](const spoofwatch::LobsterMessage&) {}),
        std::runtime_error);
    std::remove(path.c_str());
}
