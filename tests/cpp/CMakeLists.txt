include(GoogleTest)

add_executable(svllm_tests
  device_test.cpp
)
target_link_libraries(svllm_tests PRIVATE svllm GTest::gtest_main)
# Discover at test time, so building on a machine without a GPU never runs the binary.
gtest_discover_tests(svllm_tests DISCOVERY_MODE PRE_TEST)
