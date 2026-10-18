#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

namespace svllm {

// RoPE frequency scaling with rope_type "llama3".
struct Llama3RopeScaling {
  double factor;
  double low_freq_factor;
  double high_freq_factor;
  int64_t original_max_position_embeddings;
};

// Sizes and architecture flags from a Hugging Face config.json.
struct ModelConfig {
  int64_t vocab_size;
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t num_layers;
  int64_t num_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
  double rms_norm_eps;
  double rope_theta;
  bool tie_word_embeddings;
  bool qk_norm;  // Qwen3 RMSNorms Q and K per head before RoPE; Llama does not.
  std::optional<Llama3RopeScaling> rope_scaling;
};

// Reads a Qwen3 or Llama config.json. Throws on other architectures or missing fields.
ModelConfig load_model_config(const std::filesystem::path& path);

}  // namespace svllm
