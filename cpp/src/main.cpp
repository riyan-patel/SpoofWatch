#include <cstdio>

namespace spoofwatch {
int version();
}

int main() {
    std::printf("SpoofWatch scaffold — version %d\n", spoofwatch::version());
    std::printf("Feed handler / order book / feature engine not yet implemented (see README Phase 0-1).\n");
    return 0;
}
