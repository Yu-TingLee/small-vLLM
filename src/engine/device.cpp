#include "engine/device.h"

#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#ifdef SVLLM_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace svllm {

Device parse_device(std::string_view name) {
  if (name == "cuda") return Device::kCUDA;
  TORCH_CHECK(name == "cpu", "unknown device '", name, "'; expected 'cuda' or 'cpu'");
  return Device::kCPU;
}

std::string describe(Device device) {
  if (device == Device::kCPU) {
    return "CPU (" + std::to_string(at::get_num_threads()) + " threads)";
  }
#ifdef SVLLM_CUDA
  TORCH_CHECK(at::cuda::is_available(), "no CUDA device available");
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  return std::string(prop->name) + " (sm_" + std::to_string(prop->major) +
         std::to_string(prop->minor) + ")";
#else
  TORCH_CHECK(false, "built without CUDA (SVLLM_CUDA=OFF)");
#endif
}

}  // namespace svllm
