#include "model/safetensors.h"

#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/types.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "model/config.h"

namespace svllm {
namespace {

using NamedTensors = std::vector<std::pair<std::string, torch::Tensor>>;

std::string dtype_name(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kBFloat16:
      return "BF16";
    case torch::kHalf:
      return "F16";
    case torch::kFloat:
      return "F32";
    case torch::kDouble:
      return "F64";
    default:
      TORCH_CHECK(false, "no safetensors name for ", dtype);
  }
}

// Pads the header with spaces to 8 bytes, as the reference writer does.
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
    header[name] = {{"dtype", dtype_name(t.scalar_type())},
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
  void TearDown() override { std::filesystem::remove(path_); }

  // Returns the error message, checking that it names the file.
  std::string load_error() const {
    try {
      load_safetensors(path_, Device::kCPU);
    } catch (const c10::Error& e) {
      const std::string message = e.what_without_backtrace();
      EXPECT_NE(message.find(path_.string()), std::string::npos) << message;
      return message;
    }
    ADD_FAILURE() << "expected c10::Error";
    return "";
  }

  const std::filesystem::path path_ =
      std::filesystem::temp_directory_path() /
      ("svllm_" + std::to_string(std::random_device{}()) + ".safetensors");
};

class LoadSafetensors : public SafetensorsFile, public ::testing::WithParamInterface<Device> {
 protected:
  void SetUp() override {
    SafetensorsFile::SetUp();
    if (GetParam() == Device::kCUDA && !torch::cuda::is_available()) {
      GTEST_SKIP() << "no CUDA device";
    }
  }

  torch::DeviceType device_type() const {
    return GetParam() == Device::kCUDA ? torch::kCUDA : torch::kCPU;
  }
};

TEST_P(LoadSafetensors, RoundTripsBf16) {
  const NamedTensors written = {
      {"model.embed_tokens.weight", torch::randn({5, 8}).to(torch::kBFloat16)},
      {"model.norm.weight", torch::randn({8}).to(torch::kBFloat16)},
      {"scalar", torch::tensor(3.5).to(torch::kBFloat16)},
      {"empty", torch::empty({0, 4}, torch::kBFloat16)},
  };
  write_safetensors(path_, written);

  const auto loaded = load_safetensors(path_, GetParam());
  ASSERT_EQ(loaded.size(), written.size());
  for (const auto& [name, expected] : written) {
    ASSERT_TRUE(loaded.contains(name)) << name;
    const torch::Tensor& t = loaded.at(name);
    EXPECT_EQ(t.device().type(), device_type()) << name;
    EXPECT_EQ(t.scalar_type(), torch::kBFloat16) << name;
    EXPECT_EQ(t.sizes(), expected.sizes()) << name;
    EXPECT_TRUE(torch::equal(t.cpu(), expected)) << name;
  }
}

TEST_P(LoadSafetensors, ConvertsToBf16) {
  const NamedTensors written = {
      {"f16", torch::randn({2, 3}).to(torch::kHalf)},
      {"f32", torch::randn({3, 4})},
      {"f64", torch::randn({4}, torch::kDouble)},
  };
  write_safetensors(path_, written);

  const auto loaded = load_safetensors(path_, GetParam());
  for (const auto& [name, original] : written) {
    const torch::Tensor& t = loaded.at(name);
    EXPECT_EQ(t.device().type(), device_type()) << name;
    EXPECT_EQ(t.scalar_type(), torch::kBFloat16) << name;
    EXPECT_TRUE(torch::equal(t.cpu(), original.to(torch::kBFloat16))) << name;
  }
}

TEST_P(LoadSafetensors, Qwen3MatchesItsHeader) {
  const std::optional<std::filesystem::path> dir = find_qwen3_snapshot();
  if (!dir) GTEST_SKIP() << "Qwen3-0.6B not found; run scripts/download_models.sh";
  const std::filesystem::path path = *dir / "model.safetensors";

  const auto weights = load_safetensors(path, GetParam());
  const nlohmann::json header = read_header(path);
  size_t num_tensors = 0;
  for (const auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;
    ++num_tensors;
    ASSERT_TRUE(weights.contains(name)) << name;
    const torch::Tensor& t = weights.at(name);
    EXPECT_EQ(t.device().type(), device_type()) << name;
    EXPECT_EQ(t.scalar_type(), torch::kBFloat16) << name;
    EXPECT_EQ(t.sizes(), info.at("shape").get<std::vector<int64_t>>()) << name;
  }
  EXPECT_EQ(weights.size(), num_tensors);

  const ModelConfig c =
      load_model_config(std::filesystem::path(SVLLM_TEST_DATA_DIR) / "qwen3-0.6b" / "config.json");
  const std::string last = "model.layers." + std::to_string(c.num_layers - 1) + ".";
  EXPECT_EQ(weights.at("model.embed_tokens.weight").sizes(),
            torch::IntArrayRef({c.vocab_size, c.hidden_size}));
  EXPECT_EQ(weights.at(last + "self_attn.q_proj.weight").sizes(),
            torch::IntArrayRef({c.num_heads * c.head_dim, c.hidden_size}));
  EXPECT_EQ(weights.at(last + "self_attn.k_proj.weight").sizes(),
            torch::IntArrayRef({c.num_kv_heads * c.head_dim, c.hidden_size}));
  EXPECT_EQ(weights.at(last + "mlp.down_proj.weight").sizes(),
            torch::IntArrayRef({c.hidden_size, c.intermediate_size}));
}

INSTANTIATE_TEST_SUITE_P(Devices, LoadSafetensors, ::testing::Values(Device::kCPU, Device::kCUDA),
                         [](const auto& info) {
                           return info.param == Device::kCUDA ? "CUDA" : "CPU";
                         });

// A zero-copy CPU tensor must keep the mapping alive after the loader returns, and the
// copy-on-write mapping must keep in-place ops away from the file.
TEST_F(SafetensorsFile, CpuTensorOutlivesLoaderAndNeverWritesTheFile) {
  const torch::Tensor original = torch::arange(6, torch::kFloat).to(torch::kBFloat16);
  write_safetensors(path_, {{"w", original}});

  torch::Tensor w = load_safetensors(path_, Device::kCPU).at("w");
  w.add_(1);
  EXPECT_TRUE(torch::equal(w, original + 1));
  EXPECT_TRUE(torch::equal(load_safetensors(path_, Device::kCPU).at("w"), original));
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

TEST_F(SafetensorsFile, RejectsHeaderThatIsNotJson) {
  write_raw(path_, "not json", "");
  EXPECT_NE(load_error().find("parse error"), std::string::npos);
}

TEST_F(SafetensorsFile, RejectsTruncatedData) {
  write_safetensors(path_, {{"w", torch::randn({4, 4}).to(torch::kBFloat16)}});
  std::filesystem::resize_file(path_, std::filesystem::file_size(path_) - 2);
  EXPECT_NE(load_error().find("outside the"), std::string::npos);
}

TEST_F(SafetensorsFile, RejectsOffsetsOutOfRange) {
  write_raw(path_, R"({"w": {"dtype": "BF16", "shape": [32], "data_offsets": [0, 64]}})",
            std::string(8, '\0'));
  EXPECT_NE(load_error().find("data_offsets [0, 64] outside the 8-byte data buffer"),
            std::string::npos);
}

TEST_F(SafetensorsFile, RejectsOffsetsThatDisagreeWithShape) {
  write_raw(path_, R"({"w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 6]}})",
            std::string(8, '\0'));
  EXPECT_NE(load_error().find("spanning 6 bytes, but its shape and dtype need 8"),
            std::string::npos);
}

TEST_F(SafetensorsFile, RejectsUnsupportedDtype) {
  write_raw(path_, R"({"w": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}})",
            std::string(4, '\0'));
  EXPECT_NE(load_error().find("unsupported dtype I8"), std::string::npos);
}

}  // namespace
}  // namespace svllm
