#include <gtest/gtest.h>

namespace spoofwatch {
int version();
}

TEST(Scaffold, VersionCompiles) {
    EXPECT_EQ(spoofwatch::version(), 0);
}
