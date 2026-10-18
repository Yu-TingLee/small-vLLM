#include <gtest/gtest.h>

#include <filesystem>

#include "model/config.h"

namespace svllm {
namespace {

const std::filesystem::path kDataDir = SVLLM_TEST_DATA_DIR;

TEST(LoadModelConfig, Qwen3) {
  const ModelConfig c = load_model_config(kDataDir / "qwen3-0.6b" / "config.json");
  EXPECT_EQ(c.vocab_size, 151936);
  EXPECT_EQ(c.hidden_size, 1024);
  EXPECT_EQ(c.intermediate_size, 3072);
  EXPECT_EQ(c.num_layers, 28);
  EXPECT_EQ(c.num_heads, 16);
  EXPECT_EQ(c.num_kv_heads, 8);
  EXPECT_EQ(c.head_dim, 128);  // Not hidden_size / num_heads, which is 64.
  EXPECT_DOUBLE_EQ(c.rms_norm_eps, 1e-6);
  EXPECT_DOUBLE_EQ(c.rope_theta, 1e6);
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_TRUE(c.qk_norm);
  EXPECT_FALSE(c.rope_scaling.has_value());
}

TEST(LoadModelConfig, Llama32) {
  const ModelConfig c = load_model_config(kDataDir / "llama-3.2-1b-instruct" / "config.json");
  EXPECT_EQ(c.vocab_size, 128256);
  EXPECT_EQ(c.hidden_size, 2048);
  EXPECT_EQ(c.intermediate_size, 8192);
  EXPECT_EQ(c.num_layers, 16);
  EXPECT_EQ(c.num_heads, 32);
  EXPECT_EQ(c.num_kv_heads, 8);
  EXPECT_EQ(c.head_dim, 64);
  EXPECT_DOUBLE_EQ(c.rms_norm_eps, 1e-5);
  EXPECT_DOUBLE_EQ(c.rope_theta, 5e5);
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_FALSE(c.qk_norm);
  ASSERT_TRUE(c.rope_scaling.has_value());
  EXPECT_DOUBLE_EQ(c.rope_scaling->factor, 8.0);
  EXPECT_DOUBLE_EQ(c.rope_scaling->low_freq_factor, 1.0);
  EXPECT_DOUBLE_EQ(c.rope_scaling->high_freq_factor, 4.0);
  EXPECT_EQ(c.rope_scaling->original_max_position_embeddings, 8192);
}

}  // namespace
}  // namespace svllm
