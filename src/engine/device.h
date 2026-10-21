#pragma once

#include <c10/core/Device.h>

#include <string>
#include <string_view>

namespace svllm {

// The device the whole engine runs on.
enum class Device { kCUDA, kCPU };

Device parse_device(std::string_view name);

c10::Device to_torch_device(Device device);

// Human-readable name, e.g. "NVIDIA GeForce RTX 4090 (sm_89)" or "CPU (16 threads)".
std::string describe(Device device);

}  // namespace svllm
