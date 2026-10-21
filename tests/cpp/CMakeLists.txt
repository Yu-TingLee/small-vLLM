include(GoogleTest)

add_executable(svllm_tests
  device_test.cpp
  model_config_test.cpp
)
target_link_libraries(svllm_tests PRIVATE svllm GTest::gtest_main)
target_compile_definitions(svllm_tests PRIVATE SVLLM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
# Discover at test time, so building on a machine without a GPU never runs the binary.
gtest_discover_tests(svllm_tests DISCOVERY_MODE PRE_TEST)

add_executable(svllm_tokenizer_test tokenizer_test.cpp)
target_link_libraries(svllm_tokenizer_test PRIVATE svllm torch torch_cpu)
target_include_directories(svllm_tokenizer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tokenizer ${CMAKE_BINARY_DIR}/_deps/tokenizers_cpp-src/include)
add_test(NAME TokenizerTest COMMAND svllm_tokenizer_test)
set_tests_properties(TokenizerTest PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
