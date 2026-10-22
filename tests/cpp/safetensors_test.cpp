#include "model/safetensors.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/types.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "model/config.h"

namespace svllm {
namespace {

using NamedTensors = std::vector<std::pair<std::string, torch::Tensor>>;

// Pads the header with spaces to 8 bytes, as the reference writer does
void write_raw(const std::filesystem::path& path, std::string header, const std::string& data) {
  header.resize((header.size() + 7) / 8 * 8, ' ');
  const uint64_t header_size = header.size();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
  out << header << data;
}

void write_safetensors(const std::filesystem::path& path, const NamedTensors& tensors) {
  nlohmann::json header = {{"__metadata__", {{"format", "pt"}}}};
  std::string data;
  for (const auto& [name, tensor] : tensors) {
    const torch::Tensor t = tensor.contiguous();
    const uint64_t begin = data.size();
    data.append(static_cast<const char*>(t.data_ptr()), t.nbytes());
    header[name] = {{"dtype", "BF16"},
                    {"shape", t.sizes().vec()},
                    {"data_offsets", {begin, uint64_t{data.size()}}}};
  }
  write_raw(path, header.dump(), data);
}

nlohmann::json read_header(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  uint64_t header_size = 0;
  in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  std::string header(header_size, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_size));
  return nlohmann::json::parse(header);
}

std::optional<std::filesystem::path> find_qwen3_snapshot() {
  std::filesystem::path hub;
  if (const char* cache = std::getenv("HF_HUB_CACHE")) {
    hub = cache;
  } else if (const char* home = std::getenv("HF_HOME")) {
    hub = std::filesystem::path(home) / "hub";
  } else if (const char* user_home = std::getenv("HOME")) {
    hub = std::filesystem::path(user_home) / ".cache" / "huggingface" / "hub";
  }
  std::error_code ec;
  for (const auto& snapshot :
       std::filesystem::directory_iterator(hub / "models--Qwen--Qwen3-0.6B" / "snapshots", ec)) {
    if (std::filesystem::exists(snapshot.path() / "model.safetensors")) return snapshot.path();
  }
  return std::nullopt;
}

class SafetensorsFile : public ::testing::Test {
 protected:
  void SetUp() override { torch::manual_seed(0); }

  // Returns the error message, checking that it names the file
  std::string load_error() const {
    try {
      load_safetensors(path_, torch::kCPU);
    } catch (const c10::Error& e) {
      const std::string message = e.what_without_backtrace();
      EXPECT_NE(message.find(path_.string()), std::string::npos) << message;
      return message;
    }
    ADD_FAILURE() << "expected c10::Error";
    return "";
  }

  const c10::TempFile file_ = c10::make_tempfile("svllm_");
  const std::filesystem::path path_ = file_.name;
};

class LoadSafetensors : public SafetensorsFile, public ::testing::WithParamInterface<c10::Device> {
 protected:
  void SetUp() override {
    SafetensorsFile::SetUp();
    if (GetParam().is_cuda() && !torch::cuda::is_available()) {
      GTEST_SKIP() << "no CUDA device";
    }
  }

  c10::Device device() const { return GetParam(); }
};

TEST_P(LoadSafetensors, RoundTripsToBf16) {
  const NamedTensors written = {
      {"model.embed_tokens.weight", torch::randn({5, 8}).to(torch::kBFloat16)},
      {"scalar", torch::tensor(3.5).to(torch::kBFloat16)},
      {"empty", torch::empty({0, 4}, torch::kBFloat16)},
  };
  write_safetensors(path_, written);

  const auto loaded = load_safetensors(path_, device());
  ASSERT_EQ(loaded.size(), written.size());
  for (const auto& [name, original] : written) {
    ASSERT_TRUE(loaded.contains(name)) << name;
    const torch::Tensor& t = loaded.at(name);
    EXPECT_EQ(t.device().type(), device().type()) << name;
    EXPECT_EQ(t.scalar_type(), torch::kBFloat16) << name;
    EXPECT_EQ(t.sizes(), original.sizes()) << name;
    EXPECT_TRUE(torch::equal(t.cpu(), original)) << name;
  }
}

TEST_P(LoadSafetensors, Qwen3MatchesItsHeader) {
  const std::optional<std::filesystem::path> dir = find_qwen3_snapshot();
  if (!dir) GTEST_SKIP() << "Qwen3-0.6B not found; run scripts/download_models.sh";
  const std::filesystem::path path = *dir / "model.safetensors";

  const auto weights = load_safetensors(path, device());
  const nlohmann::json header = read_header(path);
  size_t num_tensors = 0;
  for (const auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;
    ++num_tensors;
    ASSERT_TRUE(weights.contains(name)) << name;
    const torch::Tensor& t = weights.at(name);
    EXPECT_EQ(t.device().type(), device().type()) << name;
    EXPECT_EQ(t.scalar_type(), torch::kBFloat16) << name;
    EXPECT_EQ(t.sizes(), info.at("shape").get<std::vector<int64_t>>()) << name;
  }
  EXPECT_EQ(weights.size(), num_tensors);

  const ModelConfig config =
      load_model_config(std::filesystem::path(SVLLM_TEST_DATA_DIR) / "qwen3-0.6b" / "config.json");
  const std::string last = "model.layers." + std::to_string(config.num_layers - 1) + ".";
  EXPECT_EQ(weights.at("model.embed_tokens.weight").sizes(),
            torch::IntArrayRef({config.vocab_size, config.hidden_size}));
  EXPECT_EQ(weights.at(last + "self_attn.q_proj.weight").sizes(),
            torch::IntArrayRef({config.num_heads * config.head_dim, config.hidden_size}));
  EXPECT_EQ(weights.at(last + "self_attn.k_proj.weight").sizes(),
            torch::IntArrayRef({config.num_kv_heads * config.head_dim, config.hidden_size}));
  EXPECT_EQ(weights.at(last + "mlp.down_proj.weight").sizes(),
            torch::IntArrayRef({config.hidden_size, config.intermediate_size}));
}

INSTANTIATE_TEST_SUITE_P(Devices, LoadSafetensors,
                         ::testing::Values(c10::Device(c10::kCPU), c10::Device(c10::kCUDA)),
                         [](const auto& info) { return info.param.is_cuda() ? "CUDA" : "CPU"; });

// CPU weights are copies, so in-place writes to either the tensor or the file never reach the other
TEST_F(SafetensorsFile, CpuTensorIsIndependentOfTheFile) {
  const torch::Tensor original = torch::arange(6, torch::kFloat).to(torch::kBFloat16);
  const torch::Tensor rewritten = torch::zeros({6}, torch::kBFloat16);
  write_safetensors(path_, {{"w", original}});

  torch::Tensor w = load_safetensors(path_, torch::kCPU).at("w");
  write_safetensors(path_, {{"w", rewritten}});
  EXPECT_TRUE(torch::equal(w, original));

  w.add_(1);
  EXPECT_TRUE(torch::equal(load_safetensors(path_, torch::kCPU).at("w"), rewritten));
}

TEST_F(SafetensorsFile, RejectsFileShorterThanHeaderLength) {
  std::ofstream(path_, std::ios::binary) << "abc";
  EXPECT_NE(load_error().find("truncated"), std::string::npos);
}

TEST_F(SafetensorsFile, RejectsHeaderLengthPastEndOfFile) {
  write_raw(path_, "{}", "");
  std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
  const uint64_t too_long = 1000;
  file.write(reinterpret_cast<const char*>(&too_long), sizeof(too_long));
  file.close();
  EXPECT_NE(load_error().find("header length 1000"), std::string::npos);
}

TEST_F(SafetensorsFile, RejectsTruncatedData) {
  write_safetensors(path_, {{"w", torch::randn({4, 4}).to(torch::kBFloat16)}});
  std::filesystem::resize_file(path_, std::filesystem::file_size(path_) - 2);
  EXPECT_NE(load_error().find("outside the"), std::string::npos);
}

TEST_F(SafetensorsFile, RejectsMisalignedTensor) {
  const std::string header = R"({"w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}} )";
  ASSERT_EQ(header.size() % 2, 1u);
  const uint64_t header_size = header.size();
  {
    std::ofstream out(path_, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out << header << std::string(2, '\0');
  }
  EXPECT_NE(load_error().find("misaligned for BF16"), std::string::npos);
}

TEST_F(SafetensorsFile, RejectsMalformedEntries) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"not json", "cannot load"},
      {R"([])", "is not a JSON object"},
      {R"({"w": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}})", "unsupported dtype I8"},
      {R"({"w": {"dtype": "BF16", "shape": [32], "data_offsets": [0, 64]}})",
       "data_offsets [0, 64] outside the 8-byte data buffer"},
      {R"({"w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 6]}})",
       "spanning 6 bytes, but its shape and dtype need 8"},
      {R"({"w": 5})", "tensor 'w'"},
      {R"({"w": {"dtype": "BF16", "shape": [-1, 2], "data_offsets": [0, 4]}})",
       "invalid shape [-1,2]"},
      {R"({"w": {"dtype": "BF16", "shape": [1.5, 2], "data_offsets": [0, 4]}})",
       "invalid shape [1.5,2]"},
      {R"({"w": {"dtype": "BF16", "shape": [4294967296, 4294967296, 4], "data_offsets": [0, 4]}})",
       "invalid shape [4294967296,4294967296,4]"},
      {R"({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4, 8]}})",
       "data_offsets [0,4,8]"},
      {R"({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
           "w": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]}})",
       "starts at byte 4"},
      {R"({"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
           "b": {"dtype": "BF16", "shape": [3], "data_offsets": [2, 8]}})",
       "starts at byte 2"},
      {R"({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})",
       "4 bytes after its last tensor"},
  };
  for (const auto& [header, fragment] : cases) {
    SCOPED_TRACE(header);
    write_raw(path_, header, std::string(8, '\0'));
    EXPECT_NE(load_error().find(fragment), std::string::npos);
  }
}

}  // namespace
}  // namespace svllm
