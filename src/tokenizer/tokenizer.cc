#include "tokenizer/tokenizer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "tokenizers_cpp.h"

namespace svllm {
Tokenizer::Tokenizer(const std::string& tokenizer_json_path) {
  // Open tokenizer.json
  std::ifstream file(tokenizer_json_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer json file: " + tokenizer_json_path);
  }

  // Load the content of tokenizer.json into string
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json_content = buffer.str();

  tokenizers_ = tokenizers::Tokenizer::FromBlobJSON(json_content);

  // Set EOS (End of Sentence) ID
  eos_id_ = tokenizers_->TokenToId("<|endoftext|>");
  if (eos_id_ == -1) {
    eos_id_ = 2;
  }
}

Tokenizer::~Tokenizer() = default;

// Turn string into Token ID array
std::vector<int32_t> Tokenizer::Encode(const std::string& text) const {
  return tokenizers_->Encode(text);
}

// Turn Token ID array back to string
std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const {
  return tokenizers_->Decode(ids);
}
}  // namespace svllm
