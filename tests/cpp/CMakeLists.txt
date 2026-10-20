include(GoogleTest)

add_executable(svllm_tests
  device_test.cpp
  model_config_test.cpp
  safetensors_test.cpp
)
target_link_libraries(svllm_tests PRIVATE svllm nlohmann_json::nlohmann_json GTest::gtest_main)
target_compile_definitions(svllm_tests PRIVATE SVLLM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
# Discover at test time, so building on a machine without a GPU never runs the binary.
gtest_discover_tests(svllm_tests DISCOVERY_MODE PRE_TEST)
