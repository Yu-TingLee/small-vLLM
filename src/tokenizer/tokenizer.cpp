#include "tokenizer/tokenizer.h"

#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>
#include <tokenizers_c.h>

#include <fstream>
#include <iterator>

namespace svllm {
namespace {

// Whether text is UTF-8 as Rust's str::from_utf8 accepts it, which tokenizers_encode unwraps:
// no overlong forms, surrogates or code points above U+10FFFF.
bool is_valid_utf8(std::string_view text) {
  size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<unsigned char>(text[i]);
    size_t len = 0;
    uint32_t code_point = 0;
    uint32_t min_code_point = 0;
    if (lead < 0x80) {
      len = 1;
      code_point = lead;
    } else if ((lead & 0xE0) == 0xC0) {
      len = 2;
      code_point = lead & 0x1F;
      min_code_point = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
      len = 3;
      code_point = lead & 0x0F;
      min_code_point = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
      len = 4;
      code_point = lead & 0x07;
      min_code_point = 0x10000;
    } else {
      return false;
    }
    if (text.size() - i < len) {
      return false;
    }

    for (size_t k = 1; k < len; ++k) {
      const auto byte = static_cast<unsigned char>(text[i + k]);
      if ((byte & 0xC0) != 0x80) {
        return false;
      }
      code_point = (code_point << 6) | (byte & 0x3F);
    }
    if (code_point < min_code_point || code_point > 0x10FFFF ||
        (code_point >= 0xD800 && code_point <= 0xDFFF)) {
      return false;
    }
    i += len;
  }

  return true;
}

}  // namespace

void Tokenizer::FreeHandle::operator()(void* handle) const {
  tokenizers_free(handle);
}

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_json) {
  std::ifstream file(tokenizer_json, std::ios::binary);
  TORCH_CHECK(file, "cannot open ", tokenizer_json.string());
  const std::string json{std::istreambuf_iterator<char>(file), {}};
  TORCH_CHECK(!file.bad(), "cannot read ", tokenizer_json.string());

  handle_.reset(tokenizers_new_from_str(json.data(), json.size()));
  size_t vocab_size = 0;
  tokenizers_get_vocab_size(handle_.get(), &vocab_size);
  vocab_size_ = static_cast<int64_t>(vocab_size);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) {
  TORCH_CHECK(handle_ != nullptr, "Tokenizer used after a move");
  // Rust would panic and abort the process instead.
  TORCH_CHECK(is_valid_utf8(text), "text to encode is not valid UTF-8");

  // Rust builds a slice from the pointer, which must not be null even when the text is empty.
  const char* data = text.empty() ? "" : text.data();

  TokenizerEncodeResult result;
  tokenizers_encode(handle_.get(), data, text.size(), /*add_special_token=*/1, &result);
  const auto free_result =
      c10::make_scope_exit([&] { tokenizers_free_encode_results(&result, 1); });

  return std::vector<int32_t>(result.token_ids, result.token_ids + result.len);
}

std::string Tokenizer::decode(std::span<const int32_t> token_ids) {
  TORCH_CHECK(handle_ != nullptr, "Tokenizer used after a move");
  for (const int32_t id : token_ids) {
    TORCH_CHECK(id >= 0 && id < vocab_size_, "token id ", id, " is outside the vocabulary [0, ",
                vocab_size_, ")");
  }

  // Rust reads the ids as u32, which the check above keeps lossless, from a non-null pointer.
  static constexpr uint32_t kNoIds = 0;
  const auto* data =
      token_ids.empty() ? &kNoIds : reinterpret_cast<const uint32_t*>(token_ids.data());

  tokenizers_decode(handle_.get(), data, token_ids.size(), /*skip_special_token=*/1);
  // The text lives in the handle until the next decode, so copy it out now.
  const char* text = nullptr;
  size_t len = 0;
  tokenizers_get_decode_str(handle_.get(), &text, &len);

  return std::string(text, len);
}

}  // namespace svllm
