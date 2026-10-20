#pragma once

#include <torch/types.h>

#include <filesystem>
#include <string>
#include <unordered_map>

#include "engine/device.h"

namespace svllm {

// Memory-maps a .safetensors file and returns its tensors by name, as bf16 on `device`.
// BF16 tensors loaded to the CPU are zero-copy views that keep the mapping alive, and
// writes to them never reach the file. Accepts BF16, F16, F32 and F64; throws c10::Error
// naming the path on other dtypes, a truncated file or an inconsistent header.
std::unordered_map<std::string, torch::Tensor> load_safetensors(const std::filesystem::path& path,
                                                                Device device);

}  // namespace svllm
