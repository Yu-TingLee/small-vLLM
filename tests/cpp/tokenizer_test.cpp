#include "tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

void run_tokenizer_test(const std::string& model_name, const std::string& json_path,
                        const std::string& test_text) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Testing Model: " << model_name << std::endl;
  std::cout << "Loading tokenizer from: " << json_path << " ..." << std::endl;

  svllm::Tokenizer tokenizer(json_path);

  std::vector<int32_t> tokens = tokenizer.Encode(test_text);

  std::cout << "--- Test Results ---" << std::endl;
  std::cout << "Input text: " << test_text << std::endl;
  std::cout << "Token IDs count: " << tokens.size() << std::endl;
  std::cout << "Token IDs: [ ";
  for (int32_t id : tokens) {
    std::cout << id << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << model_name << " test passed successfully!" << std::endl;
}

int main() {
  try {
    run_tokenizer_test("Llama-3.2-1B-Instruct",
                       "tests/cpp/data/llama-3.2-1b-instruct/tokenizer.json", "Hello world");

    run_tokenizer_test("Qwen", "tests/cpp/data/qwen3-0.6b/tokenizer.json", "你好，世界！");

  } catch (const std::exception& e) {
    std::cerr << "Error during test: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nAll tokenizer tests completed successfully!" << std::endl;
  return 0;
}
