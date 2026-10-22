#pragma once

#include <c10/core/Device.h>

#include <string>
#include <string_view>

namespace svllm {

// The device the whole engine runs on: cuda:0 or cpu.
c10::Device parse_device(std::string_view name);

// Human-readable name, e.g. "NVIDIA GeForce RTX 4090 (sm_89)" or "CPU (16 threads)".
std::string describe(c10::Device device);

}  // namespace svllm
