#include "engine/device.h"

#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#ifdef SVLLM_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace svllm {

c10::Device parse_device(std::string_view name) {
  if (name == "cuda") return c10::Device(c10::kCUDA, 0);
  TORCH_CHECK(name == "cpu", "unknown device '", name, "'; expected 'cuda' or 'cpu'");
  return c10::kCPU;
}

std::string describe(c10::Device device) {
  if (device.is_cpu()) {
    return "CPU (" + std::to_string(at::get_num_threads()) + " threads)";
  }
  TORCH_CHECK(device.is_cuda(), "unsupported device ", device);
#ifdef SVLLM_CUDA
  TORCH_CHECK(at::cuda::is_available(), "no CUDA device available");
  const cudaDeviceProp* prop = at::cuda::getDeviceProperties(device.index());
  return std::string(prop->name) + " (sm_" + std::to_string(prop->major) +
         std::to_string(prop->minor) + ")";
#else
  TORCH_CHECK(false, "built without CUDA (SVLLM_CUDA=OFF)");
#endif
}

}  // namespace svllm
