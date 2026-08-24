#include "spoofwatch/alloc_counter.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {
std::atomic<uint64_t> g_alloc_count{0};
} // namespace

namespace spoofwatch {

uint64_t alloc_count() { return g_alloc_count.load(std::memory_order_relaxed); }
void reset_alloc_count() { g_alloc_count.store(0, std::memory_order_relaxed); }

} // namespace spoofwatch

void* operator new(std::size_t size) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return ::operator new(size, std::nothrow);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
