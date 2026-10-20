#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>

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

TEST(LoadModelConfig, ErrorNamesThePath) {
  const std::filesystem::path dir = kDataDir / "qwen3-0.6b";
  try {
    load_model_config(dir);
    FAIL() << "expected c10::Error";
  } catch (const c10::Error& e) {
    EXPECT_NE(std::string(e.what()).find(dir.string()), std::string::npos) << e.what();
  }
}

TEST(LoadModelConfig, AttentionBiasIsNotImplemented) {
  std::ifstream in(kDataDir / "qwen3-0.6b" / "config.json");
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::string off = "\"attention_bias\": false";
  ASSERT_NE(text.find(off), std::string::npos);
  text.replace(text.find(off), off.size(), "\"attention_bias\": true");

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("svllm_config_" + std::to_string(std::random_device{}()) + ".json");
  std::ofstream(path) << text;
  EXPECT_THROW(load_model_config(path), c10::NotImplementedError);
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace svllm
