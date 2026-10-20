#include "model/config.h"

#include <c10/util/Exception.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace svllm {
namespace {

ModelConfig parse_model_config(const std::filesystem::path& path) {
  std::ifstream file(path);
  TORCH_CHECK(file, "cannot open ", path.string());
  const nlohmann::json j = nlohmann::json::parse(file);

  const auto model_type = j.at("model_type").get<std::string>();
  TORCH_CHECK(model_type == "qwen3" || model_type == "llama", "unsupported model_type '",
              model_type, "' in ", path.string(), "; expected 'qwen3' or 'llama'");
  for (const char* flag : {"attention_bias", "mlp_bias", "use_sliding_window"}) {
    TORCH_CHECK_NOT_IMPLEMENTED(!j.value(flag, false), flag, " is not implemented (set in ",
                                path.string(), ")");
  }

  ModelConfig config{
      .vocab_size = j.at("vocab_size").get<int64_t>(),
      .hidden_size = j.at("hidden_size").get<int64_t>(),
      .intermediate_size = j.at("intermediate_size").get<int64_t>(),
      .num_layers = j.at("num_hidden_layers").get<int64_t>(),
      .num_heads = j.at("num_attention_heads").get<int64_t>(),
      .num_kv_heads = j.at("num_key_value_heads").get<int64_t>(),
      .head_dim = j.at("head_dim").get<int64_t>(),
      .rms_norm_eps = j.at("rms_norm_eps").get<double>(),
      .rope_theta = j.at("rope_theta").get<double>(),
      .tie_word_embeddings = j.at("tie_word_embeddings").get<bool>(),
      .qk_norm = model_type == "qwen3",
      .rope_scaling = std::nullopt,
  };

  const nlohmann::json scaling = j.value("rope_scaling", nlohmann::json());
  if (!scaling.is_null()) {
    const auto rope_type = scaling.at("rope_type").get<std::string>();
    TORCH_CHECK(rope_type == "llama3", "unsupported rope_type '", rope_type, "' in ",
                path.string());
    config.rope_scaling = Llama3RopeScaling{
        .factor = scaling.at("factor").get<double>(),
        .low_freq_factor = scaling.at("low_freq_factor").get<double>(),
        .high_freq_factor = scaling.at("high_freq_factor").get<double>(),
        .original_max_position_embeddings =
            scaling.at("original_max_position_embeddings").get<int64_t>(),
    };
  }
  return config;
}

}  // namespace

ModelConfig load_model_config(const std::filesystem::path& path) {
  try {
    return parse_model_config(path);
  } catch (const c10::Error&) {
    throw;
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "cannot load ", path.string(), ": ", e.what());
  }
}

}  // namespace svllm
