#include "ops/rms_norm.h"

#include <ATen/ops/empty_like.h>
#include <c10/util/Exception.h>

namespace svllm::ops {

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
              "rms_norm: expected bf16 x and weight, got ", x.scalar_type(), " and ",
              weight.scalar_type());
  TORCH_CHECK(
      x.dim() >= 1 && weight.dim() == 1 && weight.size(0) > 0 && x.size(-1) == weight.size(0),
      "rms_norm: weight of shape ", weight.sizes(),
      " does not match the last dimension of x of shape ", x.sizes());
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
              "rms_norm: x and weight must be contiguous");
  TORCH_CHECK(x.device() == weight.device(), "rms_norm: x is on ", x.device(), " but weight is on ",
              weight.device());

  at::Tensor out = at::empty_like(x);
  switch (x.device().type()) {
    case c10::DeviceType::CPU:
      cpu::rms_norm(out, x, weight, eps);
      break;
#ifdef SVLLM_CUDA
    case c10::DeviceType::CUDA:
      cuda::rms_norm(out, x, weight, eps);
      break;
#endif
    default:
      TORCH_CHECK(false, "rms_norm: unsupported device ", x.device());
  }
  return out;
}

}  // namespace svllm::ops
