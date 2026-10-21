#pragma once

#include <torch/types.h>

#include <filesystem>
#include <string>
#include <unordered_map>

namespace svllm {

// Memory-maps a .safetensors file and returns copies of its tensors by name as bf16 on `device`,
// converting from BF16, F16, F32 or F64 and throwing c10::Error naming the path on other dtypes,
// a truncated file or a header that breaks the format.
std::unordered_map<std::string, torch::Tensor> load_safetensors(const std::filesystem::path& path,
                                                                c10::Device device);

}  // namespace svllm
