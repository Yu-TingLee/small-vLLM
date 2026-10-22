#pragma once

#include <torch/types.h>

#include <filesystem>
#include <string>
#include <unordered_map>

namespace svllm {

// Memory-maps a BF16 .safetensors file and returns copies of its tensors by name on `device`,
// throwing c10::Error naming the path on another dtype, a truncated file or a header that
// breaks the format.
std::unordered_map<std::string, torch::Tensor> load_safetensors(const std::filesystem::path& path,
                                                                c10::Device device);

}  // namespace svllm
