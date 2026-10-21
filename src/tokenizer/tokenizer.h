#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tokenizers_cpp.h"

namespace svllm {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json_path);
  ~Tokenizer();

  std::vector<int32_t> Encode(const std::string& text) const;

  std::string Decode(const std::vector<int32_t>& ids) const;

  int32_t GetEosId() const { return eos_id_; }

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizers_;
  int32_t eos_id_ = -1;
};
}  // namespace svllm
