#pragma once

#include <string>
#include <string_view>

namespace svllm {

// The device the whole engine runs on (ADR-0002).
enum class Device { kCUDA, kCPU };

Device parse_device(std::string_view name);

// Human-readable name, e.g. "NVIDIA GeForce RTX 4090 (sm_89)" or "CPU (16 threads)".
std::string describe(Device device);

}  // namespace svllm
