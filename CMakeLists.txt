cmake_minimum_required(VERSION 3.20)
project(SpoofWatch CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

option(SPOOFWATCH_BUILD_TESTS "Build GoogleTest suite" ON)

add_library(spoofwatch_core
  cpp/src/lobster_reader.cpp
  cpp/src/order_pool.cpp
  cpp/src/price_level_book.cpp
  cpp/src/order_book.cpp
  cpp/src/feature_engine.cpp
)
target_include_directories(spoofwatch_core PUBLIC cpp/include)

add_executable(spoofwatch cpp/src/main.cpp)
target_link_libraries(spoofwatch PRIVATE spoofwatch_core)

add_executable(spoofwatch_validate_book cpp/src/validate_book.cpp)
target_link_libraries(spoofwatch_validate_book PRIVATE spoofwatch_core)

add_executable(spoofwatch_features cpp/src/feature_dump.cpp)
target_link_libraries(spoofwatch_features PRIVATE spoofwatch_core)

if(SPOOFWATCH_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(spoofwatch_tests
    cpp/tests/lobster_reader_test.cpp
    cpp/tests/order_pool_test.cpp
    cpp/tests/price_level_book_test.cpp
    cpp/tests/order_book_test.cpp
    cpp/tests/ring_buffer_test.cpp
    cpp/tests/incremental_stats_test.cpp
    cpp/tests/feature_engine_test.cpp
  )
  target_link_libraries(spoofwatch_tests PRIVATE spoofwatch_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(spoofwatch_tests)
endif()
