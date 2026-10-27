#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svllm {

// A Hugging Face tokenizer, run by tokenizers-cpp's Rust library. Not thread-safe.
class Tokenizer {
 public:
  // Loads a tokenizer.json. Throws c10::Error if the file can't be read; aborts if it isn't a
  // valid tokenizer.json.
  explicit Tokenizer(const std::filesystem::path& tokenizer_json);
  Tokenizer(Tokenizer&&) = default;
  Tokenizer& operator=(Tokenizer&&) = default;

  // Same as Hugging Face's Tokenizer.encode(text). Throws c10::Error if text isn't valid UTF-8
  // or the Tokenizer was moved from.
  std::vector<int32_t> encode(std::string_view text);

  // Same as Hugging Face's Tokenizer.decode(token_ids), so special tokens are dropped. Throws
  // c10::Error on an id outside the vocabulary or if the Tokenizer was moved from.
  std::string decode(std::span<const int32_t> token_ids);

 private:
  struct FreeHandle {
    void operator()(void* handle) const;
  };

  std::unique_ptr<void, FreeHandle> handle_;  // The Rust tokenizer; null after a move.
  int64_t vocab_size_ = 0;
};

}  // namespace svllm
