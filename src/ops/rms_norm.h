#pragma once

#include <ATen/core/Tensor.h>

namespace svllm::ops {

// Definition: Root-mean-square normalization over the last dimension, scaled by a weight.
// Formula:    out = bf16(x / sqrt(mean(x^2) + eps)) * weight, with the sum accumulated in fp32.
// Domain:     x [..., hidden] and weight [hidden], contiguous bf16 on one device; hidden > 0.
// Returns:    A new bf16 tensor shaped like x, on x's device.
at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps);

// Per-device kernels; rms_norm validates the inputs and allocates out before calling one
namespace cpu {
void rms_norm(const at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps);
}
namespace cuda {
void rms_norm(const at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps);
}

}  // namespace svllm::ops
